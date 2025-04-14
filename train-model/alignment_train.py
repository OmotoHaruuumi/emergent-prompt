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
from dataloader import trainDataset,valDataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import math

def args_define():
    parser = argparse.ArgumentParser(description='SimSiam Naming Game')
    parser.add_argument('--text-enc-freeze', type=bool, default=True, metavar='tef', help='freeze text_encoder or not')
    parser.add_argument('--image-enc-freeze', type=bool, default=True, metavar='tef', help='freeze image_encoder or not')
    parser.add_argument('--use-simsiam', type=bool, default=False, metavar='us', help='use simsiam framework or not')
    parser.add_argument('--use-recon-loss',type=bool,default=True,help='use recon loss or not if not using simsiam this is True automatically')
    parser.add_argument('--use-proj', type=bool, default=False, metavar='us', help='use projctor or not')
    parser.add_argument('--clip-vision-adapter-late',type=float,default=0.2,help='clip adapter late if clip_vison_model is freeze this variavle is 0')
    parser.add_argument('--kld-loss-beta',type=float,default=0.0005,help='kld loss beta if not using kld loss this parameter is 0')
    parser.add_argument('--word-length', type=int, default=15, metavar='L', help='word dimensionality (default: 10)')
    parser.add_argument('--dictionary-size', type=int, default=50257, metavar='L', help='dictionary size (default: 100)')
    parser.add_argument('--latent-dim', type=int, default=768 ,metavar='ld', help='dimension of image encoder text encoder output')
    parser.add_argument('-hidden-dim', type=int, default=2048 ,metavar='hd', help='dimension of ')
    parser.add_argument('--epochs', type=int, default=10,metavar='N', help='No of epochs of naming game [default: 100]')
    parser.add_argument('--batch_size', type=int, default=4, metavar='N', help='batch size of model [default: 64]')
    parser.add_argument('--dataset_size', type=int, default=5000, metavar='ds', help='dataset size of model max[81783]')
    parser.add_argument('--save_every', type=int, default=1 ,metavar='se',help='number of epochs which save model [default:10]')
    parser.add_argument('--learning-rate', type=float, default=1e-5 ,metavar='LR', help='learning rate [default: 1e-3]')
    parser.add_argument('--gpt-path', type=str, default="/root/emergent-prompt/train-model/pretrained-model/trained_gpt.pt", help='directory for pretrained gpt models')
    parser.add_argument('--clip-to-gpt-path', type=str, default="/root/emergent-prompt/train-model/pretrained-model/trained_mlp.pt", help='directory for pretrained gpt adapter models')
    parser.add_argument('--translator-path', type=str, default="/root/emergent-prompt/train-model/pretrained-model/trained_translator_linear9.pt", help='directory for pretrained translator models')
    parser.add_argument('--device', type=str, default='cuda', help='device for training [mps, cuda, cpu]')
    parser.add_argument('--debug', type=bool, default=False, help='debug vs running')
    parser.add_argument('--prefix',type=str,default='trained-model',help='prefix for saved filenames')
    parser.add_argument('--out-dir', default='/root/emergent-prompt/save-output')
    parser.add_argument('--out-txt',default='/root/emergent-prompt/save-txt')
    parser.add_argument('--setting-name',default='LLMPrior')
    return parser.parse_args()


def initialize(args):
    if args.debug:
        args.dataset_size = 10
        args.epochs = 2

    runId = datetime.datetime.now().isoformat()
    experiment_dir = Path('experiments/')
    experiment_dir.mkdir(parents=True, exist_ok=True)
    runPath = mkdtemp(prefix=runId, dir=str(experiment_dir))
    print('Expt:', runPath)
    print('RunID:', runId)
    args.out_dir=os.path.join(args.out_dir,args.setting_name)
    return runPath


def negative_cosine_similarity(x, y):
        x = F.normalize(x, dim=-1)
        y = F.normalize(y, dim=-1)
        return - (x * y).sum(dim=-1).mean()

#teacher_probs must be applied log_softmax
def KLD_loss(probs,teacher_probs):
        if torch.isnan(probs).any():
            print("probs NaN detected!")
        if torch.isnan(teacher_probs).any():
            print("teacher NaN detected")
        return F.kl_div(teacher_probs,probs,reduction="batchmean")

def loss_fn(probsA,probsB,pre_latentA,pre_latentB,latentA,latentB, pre_latent_reconA,pre_latent_reconB,latent_reconA, latent_reconB,dictionary_size,device,use_sim=True,use_rec=True,beta=1.0,teacher_probsA=None,teacher_probsB=None):
    
    """
    def compute_KLD_loss(logit):
        logits_dist = torch.distributions.OneHotCategorical(probs=logit)
        prior = torch.log(torch.tensor([1.0 /dictionary_size] * dictionary_size,device=device))
        prior_dist = torch.distributions.OneHotCategorical(logits=prior.expand_as(logit))
        kl = torch.distributions.kl_divergence(logits_dist, prior_dist)
        return kl.sum(1).mean()
    """
    
    loss_sim = 0
    loss_kld = 0
    loss_recon = 0

    if use_sim:
        loss_similarityA = negative_cosine_similarity(latentA, latent_reconB)
        loss_similarityB = negative_cosine_similarity(latentB, latent_reconA)
        loss_sim = loss_similarityA/2 + loss_similarityB/2


    if beta!=0:
        loss_kldA = KLD_loss(probsA,teacher_probsA)
        loss_kldB = KLD_loss(probsB,teacher_probsB)
        loss_kld = loss_kldA/2 + loss_kldB/2
    
    
    if use_rec:
        loss_reconA = negative_cosine_similarity(pre_latentA, pre_latent_reconA)
        loss_reconB = negative_cosine_similarity(pre_latentB, pre_latent_reconB)
        loss_recon = loss_reconA/2 + loss_reconB/2


    total_loss = loss_sim + loss_recon + beta * loss_kld
    return total_loss, loss_sim, loss_recon, loss_kld

def loss_fn_single(probsA,pre_latentA, pre_latent_reconA,dictionary_size,device,use_sim=True,use_rec=True,beta=1.0,teacher_probsA=None):
    loss_sim = 0
    loss_kld = 0
    loss_recon = 0

    if beta!=0:
        loss_kld = KLD_loss(probsA,teacher_probsA)
    
    if use_rec:
        loss_recon = negative_cosine_similarity(pre_latentA, pre_latent_reconA)


    total_loss = loss_sim + loss_recon + beta * loss_kld
    return total_loss, loss_sim, loss_recon, loss_kld


def train(dataset,val_dataset,device,param_group,model,args):
    model.train()
    optimizer = optim.Adam(param_group)
    batch_size = args.batch_size
    epochs = args.epochs
    output_dir = args.out_dir
    output_prefix = args.prefix
    data_size = len(dataset)
    word_length = args.word_length
    print(f"dataset length is {data_size}")
    val_data_size = len(val_dataset)
    step_size = (data_size//batch_size)+1
    data_loader = DataLoader(dataset,batch_size=batch_size,shuffle=True)
    val_data_loader = DataLoader(val_dataset,2,shuffle=True)
    loss_history=[]
    simsiam_history=[]
    kld_history=[]
    recon_history=[]
    val_recon_history=[]
    val_sim_history=[]
    ent_history=[]
    image_collapse_history=[]
    text_collapse_history=[]

    test_data1 = dataset.get_unaugment(0)
    test_data2,test_data3 = dataset[0]
    test_data4 = dataset.get_unaugment(1)
    val_data=torch.stack([test_data1,test_data2,test_data3,test_data4])
    torch.save(val_data,os.path.join(output_dir, "validation_data.pt"),)
    val_data=val_data.to(device)
    
    for epoch in tqdm(range(epochs)):
        print("epoch"+str(epoch)+"start")
        train_loss=0
        recon_loss=0
        similarity_loss=0
        kld_loss=0
        model.eval()
        with torch.no_grad():
            pre_image_latent = model.image_encode(val_data)
            message_ids,token_embeds,probs,_ = model.text_decode(pre_image_latent)
            pre_text_latent = model.text_encode(message_ids,token_embeds)
            for i in range(len(val_data)):
                message=message_ids[i:i+1,:]
                output_list = list(message.squeeze().cpu().numpy())
                messages=model.gpt_tokenizer.decode(output_list)
                print(messages)
            print("pair image latent")
            print(-negative_cosine_similarity(pre_image_latent[1],pre_image_latent[2]).item())
            print("same image and text latent")
            print(-negative_cosine_similarity(pre_image_latent[0],pre_text_latent[0]).item())
            print("pair text latent")
            print(-negative_cosine_similarity(pre_text_latent[1],pre_text_latent[2]).item())
            print("unpair image latent")
            print(-negative_cosine_similarity(pre_image_latent[0],pre_image_latent[3]).item())
            print("different image and text latent")
            print(-negative_cosine_similarity(pre_image_latent[0],pre_text_latent[3]).item())
            print("unpair text latent")
            print(-negative_cosine_similarity(pre_text_latent[0],pre_text_latent[3]).item())
            del pre_image_latent,message_ids,token_embeds,probs,pre_text_latent

        for dataA,dataB in data_loader:
            model.train()
            optimizer.zero_grad()
            dataA=dataA.to(device)
            #dataB=dataB.to(device)
            pre_image_latentA,image_latentA,pre_text_latentA,text_latentA,probsA,teacher_probsA = model(dataA,use_proj=args.use_proj)
            #pre_image_latentB,image_latentB,pre_text_latentB,text_latentB,probsB,teacher_probsB = model(dataB,use_proj=args.use_proj)
            if args.use_simsiam:
                pre_image_latentA.detach()
                #pre_image_latentB.detach()
            """
            if args.use_proj:
                loss,loss_similarity,loss_recon,loss_kld = loss_fn(probsA,probsB,pre_image_latentA,pre_image_latentB,image_latentA,image_latentB,pre_text_latentA,pre_text_latentB,text_latentA, text_latentB,args.dictionary_size,device,use_sim=args.use_simsiam,use_rec=args.use_recon_loss, beta=args.kld_loss_beta,teacher_probsA=teacher_probsA,teacher_probsB=teacher_probsB)
            else:
                loss,loss_similarity,loss_recon,loss_kld = loss_fn(probsA,probsB,pre_image_latentA,pre_image_latentB,pre_image_latentA,pre_image_latentB,pre_text_latentA,pre_text_latentB,pre_text_latentA, pre_text_latentB,args.dictionary_size,device,use_sim=args.use_simsiam,use_rec=args.use_recon_loss, beta=args.kld_loss_beta,teacher_probsA=teacher_probsA,teacher_probsB=teacher_probsB)
            """
            loss,loss_similarity,loss_recon,loss_kld = loss_fn_single(probsA,pre_image_latentA,pre_text_latentA,args.dictionary_size,device,use_sim=args.use_simsiam,use_rec=args.use_recon_loss, beta=args.kld_loss_beta,teacher_probsA=teacher_probsA)
            loss.backward()
            optimizer.step()
            train_loss += loss
            recon_loss += loss_recon
            similarity_loss += loss_similarity
            kld_loss += loss_kld
        
        print("eval_start")
        recon=0
        simsiam=0
        entropy=0
        col_i=[]
        col_t=[]

        for unaug,aug1,aug2 in val_data_loader:
            valdata = torch.cat((unaug,aug1,aug2),dim=0)
            valdata=valdata.to(device)
            model.eval()
            with torch.no_grad():
                pre_image_latent,image_latent,pre_text_latent,text_latent,probs,teacher_probs = model(valdata,False)
            recon += (negative_cosine_similarity(pre_image_latent[0],pre_text_latent[0]).item()+negative_cosine_similarity(pre_image_latent[1],pre_text_latent[1]).item())
            
            simsiam += (negative_cosine_similarity(pre_image_latent[2],pre_text_latent[4]).item()+negative_cosine_similarity(pre_image_latent[4],pre_text_latent[2]).item())/2
            simsiam += (negative_cosine_similarity(pre_image_latent[3],pre_text_latent[5]).item()+negative_cosine_similarity(pre_image_latent[5],pre_text_latent[3]).item())/2
            
            top100_logitsA,_=torch.topk(probs[0],k=100,dim=-1)
            top100_logitsB,_=torch.topk(probs[0],k=100,dim=-1)
            probsA=F.softmax(top100_logitsA,dim=-1)
            probsB=F.softmax(top100_logitsB,dim=-1)
            entropy += ((-torch.sum(probsA*torch.log(probsA),dim=-1)).sum()).item()
            entropy += ((-torch.sum(probsB*torch.log(probsB),dim=-1)).sum()).item()
            
            pre_image_latent = F.normalize(pre_image_latent,dim=-1)
            pre_text_latent = F.normalize(pre_text_latent,dim=-1)
            col_i.append(pre_image_latent[0])
            col_i.append(pre_image_latent[1])
            col_t.append(pre_text_latent[0])
            col_t.append(pre_text_latent[1])
        
        print("recon")
        val_recon_history.append(recon/val_data_size)
        print(val_recon_history[-1])

        print("simsiam")
        val_sim_history.append(simsiam/val_data_size)
        print(val_sim_history[-1])

        print("message entropy")
        ent_history.append(entropy/(val_data_size*word_length))
        print(ent_history[-1])
        
        print("image_collapse")
        col_i = torch.stack(col_i)
        col_i = torch.std(col_i,dim=0)
        col_i = col_i.mean().item()
        image_collapse_level = max(0,1-math.sqrt(args.latent_dim)*col_i)
        image_collapse_history.append(image_collapse_level)
        print(image_collapse_history[-1])

        print("text_collapse")
        col_t = torch.stack(col_t)
        col_t = torch.std(col_t,dim=0)
        col_t = col_t.mean().item()
        text_collapse_level = max(0,1-math.sqrt(args.latent_dim)*col_t)
        text_collapse_history.append(text_collapse_level)
        print(text_collapse_history[-1])
    
        avg_loss = train_loss / step_size
        avg_recon_loss = recon_loss / step_size
        avg_similarity_loss = similarity_loss / step_size
        avg_kld_loss = kld_loss / step_size
        loss_history.append(avg_loss)
        simsiam_history.append(avg_similarity_loss)
        kld_history.append(avg_kld_loss)
        recon_history.append(avg_recon_loss)
        print(f' Avg Loss: {avg_loss:.4f}, SimSiam Loss: {avg_similarity_loss:.4f}, KLD Loss: {avg_kld_loss:.4f}, Recon Loss: {avg_recon_loss:.4f}')
        if epoch % args.save_every == 0 or epoch == epochs - 1:
            if not args.debug:
                torch.save(
                    model.state_dict(),
                    os.path.join(output_dir, f"{output_prefix}-{epoch:03d}.pt"),
                )
                print("model saved")
    model.eval()
    with torch.no_grad():
        pre_image_latent = model.image_encode(val_data)
        message_ids,token_embeds,probs,teacher_probs = model.text_decode(pre_image_latent)
        pre_text_latent = model.text_encode(message_ids,token_embeds)
        for i in range(len(val_data)):
            message=message_ids[i:i+1,:]
            output_list = list(message.squeeze().cpu().numpy())
            messages=model.gpt_tokenizer.decode(output_list)
            print(messages)
        print("pair image latent")
        print(-negative_cosine_similarity(pre_image_latent[1],pre_image_latent[2]).item())
        print("same image and text latent")
        print(-negative_cosine_similarity(pre_image_latent[0],pre_text_latent[0]).item())
        print("pair text latent")
        print(-negative_cosine_similarity(pre_text_latent[1],pre_text_latent[2]).item())
        print("unpair image latent")
        print(-negative_cosine_similarity(pre_image_latent[0],pre_image_latent[3]).item())
        print("different image and text latent")
        print(-negative_cosine_similarity(pre_image_latent[0],pre_text_latent[3]).item())
        print("unpair text latent")
        print(-negative_cosine_similarity(pre_text_latent[0],pre_text_latent[3]).item()) 

    torch.save(loss_history,os.path.join(output_dir, "loss_history.pt"),)
    torch.save(simsiam_history,os.path.join(output_dir, "simsiam_history.pt"),)
    torch.save(kld_history,os.path.join(output_dir, "kld_history.pt"),)
    torch.save(recon_history,os.path.join(output_dir, "recon_history.pt"),)
    torch.save(val_recon_history,os.path.join(output_dir, "val_recon.pt"),)
    torch.save(val_sim_history,os.path.join(output_dir, "val_sim.pt"),)
    torch.save(ent_history,os.path.join(output_dir, "val_entropy.pt"),)
    torch.save(image_collapse_history,os.path.join(output_dir, "collapse_level_image.pt"),)
    torch.save(text_collapse_history,os.path.join(output_dir, "collapse_level_text.pt"),)
    print("val_data recon loss history")
    print(val_recon_history)
    print("val_data simsiam loss history")
    print(val_sim_history)
    print("val entorpy history")
    print(ent_history)
    print("image collapse level history")
    print(image_collapse_history)
    print("text collapse level history")
    print(text_collapse_history)
    print("total loss history")
    print(loss_history)
    print("simsiam loss history")
    print(simsiam_history)
    print("kld loss history")
    print(kld_history)
    print("recon loss history ")
    print(recon_history)



def main():
    args = args_define()
    args.run_path = initialize(args) + '/'
    print(torch.cuda.memory_allocated())
    if not args.debug:
        sys.stdout = open(os.path.join(args.out_txt,args.setting_name+'-output.txt'),'w')
    device = torch.device('cuda:3')
    model = SimSiamVLM(word_length=args.word_length,latent_dim=args.latent_dim,hidden_dim=args.hidden_dim,image_enc_freeze=args.image_enc_freeze,vision_adapter_rate=args.clip_vision_adapter_late)
    model.gpt.load_state_dict(torch.load(args.gpt_path,weights_only=True))
    model.prior.load_state_dict(torch.load(args.gpt_path,weights_only=True))
    model.clip_project.load_state_dict(torch.load(args.clip_to_gpt_path,weights_only=True))
    model.translator.load_state_dict(torch.load(args.translator_path,weights_only=True))

    # モデルをLoRAトレーニング用に準備
    model.gpt = prepare_model_for_kbit_training(model.gpt)
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
    model.gpt=get_peft_model(model.gpt, lora_config) 
    model.to(device)
    dataset=trainDataset(length=args.dataset_size,augment=True)
    val_dataset = valDataset(start_index=args.dataset_size)

    param_group =[{"params":model.gpt.parameters(),"lr":args.learning_rate}]
    if not args.text_enc_freeze:
        param_group.append({"params":model.clip_text_model.parameters(),"lr":1e-6})
    if not args.image_enc_freeze:
        param_group.append({"params":model.clip_vision_adapter.parameters(),"lr":args.learning_rate})
    if args.use_proj:
        param_group.append({"params":model.projector_image.parameters(),"lr":args.learning_rate})
        param_group.append({"params":model.projector_text.parameters(),"lr":args.learning_rate})
    if not args.use_simsiam:
        args.use_recon_loss = True
    
    print(f"args:{args}")
    print(param_group)

    train(dataset,val_dataset,device, param_group,model, args)
    print("training finish!!!")
    # 終了後に標準出力を元に戻すことも可能
    sys.stdout = sys.__stdout__  # 標準出力を元に戻す

if __name__ == "__main__":
    main()

