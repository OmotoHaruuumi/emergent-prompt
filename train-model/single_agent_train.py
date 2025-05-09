import torch
import torch.nn.functional as F
import argparse
import datetime
from pathlib import Path
from SimSiamCaption import SimSiamVLM
from torch.utils.data import DataLoader
import sys
from tempfile import mkdtemp
import torch.optim as optim
from tqdm import tqdm
import os
from dataloader import trainDataset_single,valDataset_single
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import math
from Function import ImageEncoder,Adapter,TextDecoder,TeacherLLM,Translator,TextEncoder,negative_cosine_similarity
from diffusers import StableDiffusionPipeline,AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from diffusers.optimization import get_scheduler


def args_define():
    parser = argparse.ArgumentParser(description='Unsupervised VLM training')
    parser.add_argument('--use_diffusion',type=bool,default=True,help='reconstruct image or not')
    parser.add_argument('--use_prior',type=bool,default=False,help='use prior distribution or not')
    parser.add_argument('--kld_loss_beta',type=float,default=0.0005,help='kld loss beta if not using kld loss this parameter is 0')
    parser.add_argument('--LLM_prior',type=bool,default=True,help='prior distribution : LLM or Uniform distribution')
    parser.add_argument('--LLM_mode',type=str,default="all",help="set LM mode,you can select from [all,uni,bi]")
    parser.add_argument('--word_length', type=int, default=20, metavar='L', help='max word length ')
    parser.add_argument('--dictionary_size', type=int, default=50257, metavar='L', help='dictionary size (default: 100)')
    parser.add_argument('--latent_dim', type=int, default=768 ,metavar='ld', help='dimension of image encoder text encoder output')
    parser.add_argument('--prefix_length', type=int, default=10 , help='gpt prefix length')
    parser.add_argument('--epochs', type=int, default=3,metavar='N', help='No of epochs of naming game [default: 100]')
    parser.add_argument('--batch_size', type=int, default=24, metavar='N', help='batch size of model [default: 64]')
    parser.add_argument('--dataset_size', type=int, default=100, metavar='ds', help='dataset size of model max[81783]')
    parser.add_argument('--save_every', type=int, default=2 ,metavar='se',help='number of epochs which save model [default:10]')
    parser.add_argument('--learning_rate', type=float, default=1e-5 ,metavar='LR', help='learning rate [default: 1e-3]')
    parser.add_argument('--gpt_path', type=str, default="/root/emergent-prompt/train-model/pretrained-model/trained_gpt.pt", help='directory for pretrained gpt models')
    parser.add_argument('--clip_to_gpt_path', type=str, default="/root/emergent-prompt/train-model/pretrained-model/trained_mlp.pt", help='directory for pretrained gpt adapter models')
    parser.add_argument('--translator_path', type=str, default="/root/emergent-prompt/train-model/pretrained-model/trained_translator_linear9.pt", help='directory for pretrained translator models')
    parser.add_argument('--stable_diffusion_model_name',type=str,default="CompVis/stable-diffusion-v1-4")
    parser.add_argument('--debug', type=bool, default=True, help='debug vs running')
    parser.add_argument('--prefix',type=str,default='trained-model',help='prefix for saved filenames')
    parser.add_argument('--out_dir', default='/root/emergent-prompt/save-output')
    parser.add_argument('--out_txt',default='/root/emergent-prompt/save-txt')
    parser.add_argument('--setting_name',default='test')
    return parser.parse_args()


def main():
    args=args_define()
    if not args.debug:
        sys.stdout = open(os.path.join(args.out_txt,args.setting_name+'-output.txt'),'w')
    device = torch.device('cuda:3')

    image_encoder = ImageEncoder()
    adapter = Adapter((args.latent_dim,(args.latent_dim*args.prefix_length)//2,args.latent_dim*args.prefix_length,))
    text_decoder=TextDecoder(max_length=args.word_length)
    prior_decoder = TeacherLLM(max_length=args.word_length,mode=args.LLM_mode)
    translator=Translator(latent_dim=args.latent_dim)
    text_encoder = TextEncoder()

    adapter.load(args.clip_to_gpt_path)
    text_decoder.load(args.gpt_path)
    prior_decoder.load(args.gpt_path)
    translator.load(args.translator_path)

    # モデルをLoRAトレーニング用に準備
    text_decoder.model = prepare_model_for_kbit_training(text_decoder.model)
    #LoRAの設定を定義
    lora_config = LoraConfig(
        r=16,                          # 更新行列のランク
        lora_alpha=16,                 # スケーリング係数
        target_modules=["attn.c_attn", "attn.c_proj"],  #学習する層    
        lora_dropout=0.1,              # ドロップアウト率
        bias="none",                   # バイアスの扱い
        task_type="CAUSAL_LM"          # タスクの種類
        )
    # モデルをLoRAでラップ
    text_decoder.model=get_peft_model(text_decoder.model, lora_config) 

    image_encoder.requires_grad_(False)
    adapter.requires_grad_(False)
    prior_decoder.requires_grad_(False)

    image_encoder.to(device)
    adapter.to(device)
    text_decoder.to(device)
    translator.to(device)
    text_encoder.to(device)
    if args.LLM_prior:
        prior_decoder.to(device)

    param_group =[{"params":text_decoder.parameters(),"lr":args.learning_rate}]
    print(param_group)
    optimizer = optim.Adam(param_group)



    if args.use_diffusion:
        pipe = StableDiffusionPipeline.from_pretrained(
            args.stable_diffusion_model_name
            )
        vae = pipe.vae
        unet = pipe.unet
        vae.to(device)
        unet.to(device)
        noise_scheduler=pipe.scheduler
    
    dataset=trainDataset_single(length=100)
    val_dataset=valDataset_single(length=100)

    data_size = len(dataset)
    print(f"dataset length is {data_size}")
    step_size = (data_size//args.batch_size)+1

    dataloader = DataLoader(dataset,batch_size=args.batch_size,shuffle=True) 
    val_dataloader = DataLoader(val_dataset,batch_size=args.batch_size,shuffle=True) 

    loss_history=[]
    recon_history=[]
    kld_history=[]
    val_recon_history=[]

    for epoch in tqdm(range(args.epochs)):
        print("epoch "+str(epoch)+" start")
        train_loss=0.0
        recon_loss=0.0
        kld_loss=0.0
        for data ,sd_data in dataloader:
            text_decoder.train()
            data=data.to(device)
            image_embed = image_encoder(data)
            prefix=adapter(image_embed)
            message_ids,caption_emebeds,probs = text_decoder(prefix)
            transalated_embeds = translator(message_ids,caption_emebeds)
            if args.use_prior:
                if args.LLM_prior:
                    log_prior_probs=prior_decoder(probs,message_ids,caption_emebeds)
                else:
                    _, _, vocab_size = probs.shape
                    prior_probs = torch.full_like(probs,1.0/vocab_size)
                    log_prior_probs = prior_probs.log()
                kld = F.kl_div(log_prior_probs,probs,reduction="batchmean")
                loss_kld = kld * args.kld_loss_beta
            else:
                loss_kld=torch.tensor(0.0)
            if args.use_diffusion:
                sd_data=sd_data.to(device)
                image_latent = vae.encode(sd_data).latent_dist.sample()
                image_latent = image_latent * 0.18215

                noise = torch.randn_like(image_latent)
                bsz = image_latent.shape[0]    
                timesteps = torch.randint(0,noise_scheduler.config.num_train_timesteps,(bsz,),device=image_latent.device)
                timesteps=timesteps.long()

                noisy_latents = noise_scheduler.add_noise(image_latent, noise, timesteps)
                encoder_hidden_state = text_encoder(transalated_embeds,apply_projection=False)
                target = noise
                model_pred=unet(noisy_latents,timesteps,encoder_hidden_state).sample
                loss_recon = F.mse_loss(model_pred,target,reduction="mean")
            else:
                text_embed = text_encoder(transalated_embeds,apply_projection=True)
                loss_recon = negative_cosine_similarity(image_embed,text_embed)
            
            loss = loss_kld + loss_recon
            train_loss += loss.item()
            kld_loss +=loss_kld.item()
            recon_loss += loss_recon.item()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        loss_history.append(train_loss/step_size)
        kld_history.append(kld_loss/step_size)
        recon_history.append(recon_loss/step_size)
        print(f' Avg Loss: {loss_history[-1]:.4f}, KLD Loss: {kld_history[-1]:.4f}, Recon Loss: {recon_history[-1]:.4f}')
        if epoch % args.save_every == 0 or epoch == args.epochs - 1:
            if not args.debug:
                text_decoder.model.save_pretrained(os.path.join(args.out_dir,f"{args.prefix}-lora-{epoch:03d}"))
                print("LoRA adapter saved")
    
    if not args.debug:
        torch.save(loss_history,os.path.join(args.out_dir, "loss_history.pt"),)
        torch.save(kld_history,os.path.join(args.out_dir, "kld_history.pt"),)
        torch.save(recon_history,os.path.join(args.out_dir, "recon_history.pt"),)
    print("total loss history")
    print(loss_history)
    print("kld loss history")
    print(kld_history)
    print("recon loss history ")
    print(recon_history)

if __name__ == "__main__":
    main()