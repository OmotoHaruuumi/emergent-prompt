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
    def forward(self,x):
        return self.model(x)
    
#画像エンコーダー＋テキストデコーダーモデル
class SimSiamVLM(nn.Module):
    #prefix_size means output of image encoder
    def __init__(self,word_length=15,latent_dim=768,hidden_dim=2048,image_enc_freeze=True,vision_adapter_rate=0.0,prefix_length=10,prefix_size=768):
        super(SimSiamVLM,self).__init__()
        self.prefix_length = prefix_length
        self.word_length = word_length 
        self.latent_dim=latent_dim
        self.hidden_dim=hidden_dim
        self.image_enc_freeze= image_enc_freeze
        self.clip_vision_model = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-large-patch14")
        for param in self.clip_vision_model.parameters():
            param.requires_grad = False
        self.clip_vision_adapter = ClipVisionAdapter(ratio=vision_adapter_rate)
        self.projector_image = nn.Sequential(nn.Linear(self.latent_dim,self.hidden_dim, bias=False),
                                        nn.BatchNorm1d(self.hidden_dim),
                                        nn.ReLU(inplace=True), # first layer
                                        nn.Linear(self.hidden_dim,self.hidden_dim, bias=False),
                                        nn.BatchNorm1d(self.hidden_dim))
        self.projector_text =  nn.Sequential(nn.Linear(self.latent_dim, self.hidden_dim, bias=False),
                                        nn.ReLU(inplace=True), # first layer
                                        nn.Linear(self.hidden_dim,self.hidden_dim, bias=False))
        self.clip_text_model = MyCLIPTextModel("openai/clip-vit-large-patch14")
        self.gpt = GPT2LMHeadModel.from_pretrained("gpt2")
        self.prior = GPT2LMHeadModel.from_pretrained("gpt2")
        self.gpt_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.gpt_embedding_size = self.gpt.transformer.wte.weight.shape[1] #768 in case of "gpt2"
        self.clip_project = MLP((prefix_size,(self.gpt_embedding_size*prefix_length)//2,self.gpt_embedding_size*prefix_length,))
        self.bos_embed,self.eos_embed = embed_special_token()
        self.translator = torch.nn.Linear(self.latent_dim,self.latent_dim)
    def image_encode(self,image):
        with torch.no_grad():
            image_latent = self.clip_vision_model(image).image_embeds #image_latent=(batch,768)
        if self.image_enc_freeze:
            return image_latent
        else:
            adapted_features = self.clip_vision_adapter(image_latent)
            return adapted_features
    def text_decode(self,image_latent):
        max_length=self.word_length
        #image_latent is (5,768)
        #prefix_embed is (5,10,768),この768はgpt2の潜在空間が768,入力の次元の768とは無関係
        prefix_embeds = self.clip_project(image_latent).view(-1,self.prefix_length,self.gpt_embedding_size)
        probs = []
        teacher_probs = []
        #単語から埋め込みに変換する行列
        gpt2_wte = self.gpt.transformer.wte.weight
        messages_ids = None
        for i in range(max_length):
            outputs = self.gpt(inputs_embeds=prefix_embeds)
            with torch.no_grad():
                if i ==0:
                    teacher_outputs = self.prior(inputs_embeds=prefix_embeds)
                else:
                    teacher_outputs = self.prior(inputs_embeds=prefix_embeds[:,10:,]) 
            logit = outputs.logits[:,-1,:] #最後の一文字に続く単語の確率分布を予測する
            teacher_logit = teacher_outputs.logits[:,-1,:]
            logit = logit - logit.max(dim=-1, keepdim=True)[0]
            teacher_logit = teacher_logit - teacher_logit.max(dim=-1, keepdim=True)[0]
            if self.training:
                z_sampled_soft = gumbel_softmax(logit,1.0)
            else:
                z_sampled_soft = F.softmax(logit,dim=-1)
            teacher_soft = F.log_softmax(teacher_logit,dim=-1)
            probs.append(z_sampled_soft)    
            teacher_probs.append(teacher_soft)
            z_sampled_onehot, next_word = straight_through_discretize(z_sampled_soft)
            #gpt2_wteは語彙数×データの次元の形状をした行列,next_token_embed[batch,1,768]
            next_token_embed = (z_sampled_onehot @  gpt2_wte).unsqueeze(1)
            if messages_ids is None:
                messages_ids = next_word
            else:
                messages_ids = torch.cat((messages_ids,next_word),dim=1)
            prefix_embeds = torch.cat((prefix_embeds,next_token_embed),dim=1)
        #最初の10要素は画像の埋め込み表現なので除外する
        outputs_embeds = prefix_embeds[:,10:,]
        probs = torch.stack(probs).permute(1,0,2)
        teacher_probs = torch.stack(teacher_probs).permute(1,0,2)
        return messages_ids,outputs_embeds,probs,teacher_probs
    def text_encode(self,message_ids,caption_embeds,padding_index=220):
        batch=caption_embeds.size(0)
        seq_length=caption_embeds.size(1)
        bos_embed = self.bos_embed.expand(batch,1,self.latent_dim).to(device=caption_embeds.device)
        eos_embed = self.eos_embed.to(device=caption_embeds.device)
        match = (message_ids== padding_index)
        caption_embeds[match] = eos_embed
        eos_embed=eos_embed.expand(batch,1,self.latent_dim)
        caption_embeds_add_spacial_token = torch.cat([bos_embed,caption_embeds,eos_embed],dim=1)
        gpt_embeds = caption_embeds_add_spacial_token.reshape(-1,self.latent_dim)
        clip_embeds = self.translator(gpt_embeds)
        #bosとeosを足したので長さが2増えてる
        clip_embeds = clip_embeds.reshape(batch,seq_length+2,self.latent_dim)
        output_embeds = self.clip_text_model(clip_embeds).text_embeds
        return output_embeds
    def forward(self,data,use_proj=True):
        pre_image_latent = self.image_encode(data)
        message_ids,token_embeds,probs,teacher_probs = self.text_decode(pre_image_latent)
        pre_text_latent = self.text_encode(message_ids,token_embeds)
        if use_proj:
            image_latent = self.projector_image(pre_image_latent.detach())
            text_latent = self.projector_text(pre_text_latent)
        else:
            image_latent = torch.zeros(0)
            text_latent = torch.zeros(0)
        return pre_image_latent,image_latent,pre_text_latent,text_latent,probs,teacher_probs
    #return list(message) for example: ["good morning","be careful","three people standing"]
    def get_mesaages(self,image_data):
        self.eval()
        messages = []
        with torch.no_grad():
            image_latent = self.image_encode(image_data)
            message_ids,token_embeds,logits = self.text_decode(image_latent)
            output_list = list(message_ids.squeeze().cpu().numpy())
            messages=self.gpt_tokenizer.decode(output_list)
        return messages