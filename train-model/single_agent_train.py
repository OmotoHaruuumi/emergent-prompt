import torch
import torch.nn.functional as F
import argparse
from torchmetrics.multimodal import CLIPScore
import time
from pathlib import Path
from torch.utils.data import DataLoader
import sys
import torch.optim as optim
from tqdm import tqdm
import os
from dataloader import trainDataset_single,valDataset_single
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from Function import ImageEncoder,Adapter,TextDecoder,TeacherLLM,Translator,TextEncoder,negative_cosine_similarity
from diffusers import StableDiffusionPipeline
from diffusers.optimization import get_scheduler
from transformers import GPT2Tokenizer
import torch.cuda
from PIL import Image

def args_define():
    parser = argparse.ArgumentParser(description='Unsupervised VLM training')
    parser.add_argument('--use_diffusion',type=bool,default=False,help='reconstruct image or not')
    parser.add_argument('--use_prior',type=bool,default=False,help='use prior distribution or not')
    parser.add_argument('--kld_loss_beta',type=float,default=0.0002,help='kld loss beta if not using kld loss this parameter is 0')
    parser.add_argument('--LLM_prior',type=bool,default=True,help='prior distribution : LLM or Uniform distribution')
    parser.add_argument('--LLM_mode',type=str,default="all",help="set LM mode,you can select from [all,uni,bi]")
    parser.add_argument('--word_length', type=int, default=20, metavar='L', help='max word length ')
    parser.add_argument('--dictionary_size', type=int, default=50257, metavar='L', help='dictionary size (default: 100)')
    parser.add_argument('--latent_dim', type=int, default=768 ,metavar='ld', help='dimension of image encoder text encoder output')
    parser.add_argument('--prefix_length', type=int, default=10 , help='gpt prefix length')
    parser.add_argument('--epochs', type=int, default=1,metavar='N', help='No of epochs of naming game [default: 100]')
    parser.add_argument('--batch_size', type=int, default=4, metavar='N', help='batch size of model [default: 64]')
    parser.add_argument('--dataset_size', type=int, default=100, metavar='ds', help='dataset size of model max[81783]')
    parser.add_argument('--save_every', type=int, default=1 ,metavar='se',help='number of epochs which save model [default:10]')
    parser.add_argument('--learning_rate', type=float, default=5e-6 ,metavar='LR', help='learning rate [default: 1e-5]')
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
    print(f"args:{args}")
    device = torch.device('cuda:3')

    image_encoder = ImageEncoder()
    adapter = Adapter((args.latent_dim,(args.latent_dim*args.prefix_length)//2,args.latent_dim*args.prefix_length,))
    text_decoder=TextDecoder(max_length=args.word_length)
    prior_decoder = TeacherLLM(max_length=args.word_length,mode=args.LLM_mode)
    translator=Translator(latent_dim=args.latent_dim)
    text_encoder = TextEncoder()
    gpt_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    adapter.load(args.clip_to_gpt_path)
    text_decoder.load(args.gpt_path)
    translator.load(args.translator_path)

    # モデルをLoRAトレーニング用に準備
    text_decoder.model = prepare_model_for_kbit_training(text_decoder.model)
    #LoRAの設定を定義
    lora_config = LoraConfig(
        r=16,                                       # 更新行列のランク
        lora_alpha=16,                               # スケーリング係数
        target_modules=["attn.c_attn", "attn.c_proj"], #学習する層
        lora_dropout=0.1,                              # ドロップアウト率
        bias="none",                                 # バイアスの扱い
        task_type="CAUSAL_LM"                        # タスクの種類
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
        prior_decoder.load(args.gpt_path)
        prior_decoder.to(device)
    clip_score_metric = CLIPScore().to(device)

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

    dataset=trainDataset_single(length=args.dataset_size)
    val_dataset=valDataset_single(length=10)
    val_data1,val_data2,val_data3 = val_dataset.get_samples()
    val_images = torch.stack([val_data1,val_data2,val_data3])
    val_images=val_images.to(device)


    data_size = len(dataset)
    print(f"dataset length is {data_size}")
    step_size = (data_size//args.batch_size)+1

    dataloader = DataLoader(dataset,batch_size=args.batch_size,shuffle=True)
    val_dataloader = DataLoader(val_dataset,batch_size=args.batch_size,shuffle=True)

    loss_history=[]
    recon_history=[]
    kld_history=[]
    val_score_history=[]
    inference_time=[]
    used_memory=[]

    for epoch in tqdm(range(args.epochs)):
        print(f"epoch {epoch} start")
        text_decoder.train()
        train_loss=0.0
        recon_loss=0.0
        kld_loss=0.0
        epoch_start_time = time.time()
        beta = args.kld_loss_beta * min(1.0,(epoch+1)/args.epochs)

        for i, (data ,sd_data) in enumerate(dataloader):
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
                loss_kld = kld * beta
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


        epoch_end_time = time.time()
        epoch_time = epoch_end_time - epoch_start_time
        epoch_memory = torch.cuda.max_memory_allocated(device=device) / (1024**3)

        inference_time.append(epoch_time)
        used_memory.append(epoch_memory)

        loss_history.append(train_loss/step_size)
        kld_history.append(kld_loss/step_size)
        recon_history.append(recon_loss/step_size)
        print(f' Avg Loss: {loss_history[-1]:.4f}, KLD Loss: {kld_history[-1]:.4f}, Recon Loss: {recon_history[-1]:.4f}, Epoch Time: {epoch_time:.2f}s, Epoch Memory Used: {epoch_memory:.2f} MB')
        if epoch % args.save_every == 0 or epoch == args.epochs - 1:
            if not args.debug:
                text_decoder.model.save_pretrained(os.path.join(args.out_dir,f"{args.prefix}-lora-{epoch:03d}"))
                print("LoRA adapter saved")

        print("eval start")
        # CLIPScoreの初期化
        clip_score_metric.reset()
        text_decoder.eval()
        with torch.no_grad():
            for val_data in val_dataloader:
                val_data = val_data.to(device)
                images_for_clip_score = (val_data.clamp(0, 1) * 255).to(torch.uint8)
                image_embed = image_encoder(val_data)
                prefix = adapter(image_embed)
                message_ids,_,_ = text_decoder(prefix)
                generated_captions_batch = []
                for i in range(len(message_ids)):
                    message = message_ids[i:i+1,:]
                    output_list = list(message.squeeze().cpu().numpy())
                    messages=gpt_tokenizer.decode(output_list)
                    generated_captions_batch.append(messages)
                clip_score_metric.update(images_for_clip_score, generated_captions_batch)
        final_avg_clip_score_tensor = clip_score_metric.compute()
        final_avg_clip_score = final_avg_clip_score_tensor.item()
        val_score_history.append(final_avg_clip_score)
        print(f' Validation Score: {val_score_history[-1]:.4f}')

        with torch.no_grad():
            image_embed = image_encoder(val_images)
            prefix = adapter(image_embed)
            message_ids,_,_ = text_decoder(prefix)
            for i in range(len(message_ids)):
                message = message_ids[i:i+1,:]
                output_list = list(message.squeeze().cpu().numpy())
                messages=gpt_tokenizer.decode(output_list)
                print("generated message")
                print(messages)

    if not args.debug:
        torch.save(loss_history,os.path.join(args.out_dir,args.setting_name, "loss_history.pt"),)
        torch.save(kld_history,os.path.join(args.out_dir,args.setting_name, "kld_history.pt"),)
        torch.save(recon_history,os.path.join(args.out_dir,args.setting_name, "recon_history.pt"),)
        torch.save(val_score_history,os.path.join(args.out_dir,args.setting_name, "val_history.pt"),)
        torch.save(inference_time,os.path.join(args.out_dir,args.setting_name, "epoch_time.pt"),)
        torch.save(used_memory,os.path.join(args.out_dir,args.setting_name, "memory_use.pt"),)
    print("total loss history")
    print(loss_history)
    print("kld loss history")
    print(kld_history)
    print("recon loss history ")
    print(recon_history)
    print("val history ")
    print(val_score_history)
    print("training time history")
    print(inference_time)
    print("memory used history")
    print(used_memory)

if __name__ == "__main__":
    main()