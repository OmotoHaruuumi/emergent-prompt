import torch
import torch.nn as nn
from torch.nn import functional as nnf
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from enum import Enum
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW, get_linear_schedule_with_warmup
from transformers import CLIPVisionModelWithProjection
from tqdm import tqdm
import os
import pickle
import sys
import argparse
import json
from typing import Tuple, Optional, Union
from utils import Logger, TransformerModel,gumbel_softmax, straight_through_discretize,MyCLIPTextModel,embed_special_token,ClipVisionAdapter

#画像エンコーダーとテキストデコーダーを接続
class MLP(nn.Module):
    def __init__(self,sizes,bias=True,act=nn.Tanh):
        super(MLP,self).__init__()
        layers = []
        for i in range(len(sizes)-1):
            layers.append(nn.Linear(sizes[i],sizes[i+1],bias=bias))
            if i < len(sizes) - 2:
                layers.append(act())
        self.model = nn.Sequential(*layers)
        self.prefix_length=10
        self.gpt_embeddig_size=768
    def load(self,url):
        self.model.load_state_dict(torch.load(url,weight_only=True))
    def forward(self,x):
        prefix_embeds=self.model(x).view(-1,self.prefix_length,self.gpt_embedding_size)
        return prefix_embeds
    
class ImageEncoder(nn.Module):
    def __init__(self):
        super(ImageEncoder).__init__()
        self.model = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-large-patch14")
    def forward(self,image):
        with torch.no_grad():
            image_latent = self.model(image).image_embeds
        return image_latent

class TextDecoder(nn.Module):
    def __init__(self,max_length):
        super(TextDecoder).__init__()
        self.model = GPT2LMHeadModel.from_pretrained("gpt2")
        self.max_length=max_length
        self.weight = self.model.transformer.wte.weight
    def load(self,url):
        self.model.load_state_dict(torch.load(url,weight_only=True))
    def forward(self,prefix_embeds):
        probs = []
        for i in range(self.ax_length):
            outputs = self.model(inputs_embeds=prefix_embeds)
            logit = outputs.logits[:,-1,:] #最後の一文字に続く単語の確率分布を予測する
            logit = logit - logit.max(dim=-1, keepdim=True)[0]
            #teacher logit min is minus because teacher max became 0
            if i==0:
                logit[:,messages_ids] = logit.min(dim=-1, keepdim=True)[0]
            if self.training:
                z_sampled_soft = gumbel_softmax(logit,1.0)
            else:
                z_sampled_soft = F.softmax(logit,dim=-1)
            probs.append(z_sampled_soft)    
            z_sampled_onehot, next_word = straight_through_discretize(z_sampled_soft)
            #gpt2_wteは語彙数×データの次元の形状をした行列,next_token_embed[batch,1,768]
            next_token_embed = (z_sampled_onehot @  self.weight).unsqueeze(1)
            if messages_ids is None:
                messages_ids = next_word
            else:
                messages_ids = torch.cat((messages_ids,next_word),dim=1)
            prefix_embeds = torch.cat((prefix_embeds,next_token_embed),dim=1)
        #最初の10要素は画像の埋め込み表現なので除外する
        outputs_embeds = prefix_embeds[:,10:,]
        probs = torch.stack(probs).permute(1,0,2)
        return messages_ids,outputs_embeds,probs

class Translator(nn.Module):
    def __init__(self,latent_dim):
        super(Translator).__init__()
        self.model = torch.nn.Linear(latent_dim,latent_dim)
        self.bos_embed,self.eos_embed = embed_special_token()
        self.padding_embeds = 220
    def load(self,url):
        self.model.load_state_dict(torch.load(url,weight_only=True))
    def forward(self,message_ids,caption_embeds):
        batch=caption_embeds.size(0)
        seq_length=caption_embeds.size(1)
        bos_embed = self.bos_embed.expand(batch,1,self.latent_dim).to(device=caption_embeds.device)
        eos_embed = self.eos_embed.to(device=caption_embeds.device)
        match = (message_ids== self.padding_index)
        caption_embeds[match] = eos_embed
        eos_embed=eos_embed.expand(batch,1,self.latent_dim)
        caption_embeds_add_spacial_token = torch.cat([bos_embed,caption_embeds,eos_embed],dim=1)
        gpt_embeds = caption_embeds_add_spacial_token.reshape(-1,self.latent_dim)
        clip_embeds = self.translator(gpt_embeds)
        #bosとeosを足したので長さが2増えてる
        clip_embeds = clip_embeds.reshape(batch,seq_length+2,self.latent_dim)
        return clip_embeds

class TeacherLLM(nn.Module):
    def __init__(self,max_length,uni=False,Bi=False):
        super(TeacherLLM).__init__()
        self.model = GPT2LMHeadModel.from_pretrained("gpt2")
        self.max_length=max_length
        self.mode="all"
        if uni==True:
            self.mode="uni"
        if Bi ==True:
            self.mode="bi"
    def load(self,url):
        self.model.load_state_dict(torch.load(url,weight_only=True))
    def forward(self,probs,message_ids,caption_embeds):
        teacher_probs = []
        with torch.no_grad():
            for i in range(self.max_length):
                if i==0:
                    teacher_outputs = probs[:,0,:]
                elif i==1:
                    teacher_outputs = self.mdoel(inputs_embeds=caption_embeds[:,:i,:])
                else:
                    if self.mode=="all":
                        teacher_outputs = self.model(inputs_embeds=caption_embeds[:,:i,:])
                    if self.mode=="uni":
                        teacher_outputs = self.model(inputs_embeds=caption_embeds[:,i-1:i,:])
                    if self.mode=="bi":
                        teacher_outputs = self.model(inputs_embeds=caption_embeds[:,i-2:i,:])
                teacher_logit=teacher_outputs.logits[:,-1,:]
                teacher_logit = teacher_logit - teacher_logit.max(dim=-1, keepdim=True)[0]
                if i !=0:
                    teacher_logit[:,message_ids[i-1]] = teacher_logit.min(dim=-1, keepdim=True)[0]
                teacher_soft = F.log_softmax(teacher_logit,dim=-1)
                teacher_probs.append(teacher_soft)
            teacher_probs = torch.stack(teacher_probs).permute(1,0,2)
            return teacher_probs

class TextEncoder(nn.Module):
    def __init__(self):
        super(TextEncoder).__init__()
        self.model = MyCLIPTextModel("openai/clip-vit-large-patch14")
    def forward(self,translated_embeds,apply_projection=True):
        output_embeds = self.model(translated_embeds)
        if apply_projection:
            return output_embeds.text_embeds
        else:
            return output_embeds.last_hidden_state


