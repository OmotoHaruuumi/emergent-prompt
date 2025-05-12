import torch
from torch.utils.data import Dataset
from PIL import Image,ImageFilter
from torchvision import transforms
from datasets import load_dataset
import random

class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""
    def __init__(self, sigma=None):
        if sigma is None:
            sigma = [.1, 2.]
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

class trainDataset(Dataset):
    def __init__(self,image_size=224,length=5000,augment=True):
        length=min(81783,length)
        length=max(2,length)
        ds = load_dataset("jpawan33/fkr30k-image-captioning-dataset")
        self.ds=ds["train"]
        self.image_size=image_size
        self.augment=augment
        self.data=[]
        if self.augment:
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(self.image_size,scale=(0.8,1.0)),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                GaussianBlur(sigma=[0.2, 2.0]),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
                ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(self.image_size),
                transforms.CenterCrop(self.image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5])
            ])
        self.unaugment_transform = transforms.Compose([
                transforms.Resize(self.image_size),
                transforms.CenterCrop(self.image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5])  # grayscale
                ])
        for idx in range(length):
            self.data.append(self.ds[idx]["image"].convert("RGB"))
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        # データ1とデータ2をセットで返す
        img1 = self.transform(self.data[idx])
        img2 = self.transform(self.data[idx])
        return img1, img2
    def get_unaugment(self,idx):
        return self.unaugment_transform(self.data[idx])


class valDataset(Dataset):
    def __init__(self,image_size=224,start_index=5000,length=1000):
        length=min(82783-start_index,length)
        length=max(2,length)
        ds = load_dataset("jpawan33/fkr30k-image-captioning-dataset")
        self.ds=ds["train"]
        self.image_size=image_size
        self.data=[]
        #本当はtransformとaug_transformは逆だが後で直す
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(self.image_size,scale=(0.8,1.0)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            GaussianBlur(sigma=[0.2, 2.0]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
            ])
        self.aug_transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.CenterCrop(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])  # grayscale
            ])
        for idx in range(length):
            idx=idx+start_index
            self.data.append(self.ds[idx]["image"].convert("RGB"))
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        # データ1とデータ2をセットで返す
        img1 = self.transform(self.data[idx])
        aug_image1 = self.aug_transform(self.data[idx])
        aug_image2 = self.aug_transform(self.data[idx])
        return img1 , aug_image1 ,aug_image2

class trainDataset_single(Dataset):
    def __init__(self,image_size=224,sd_image_size=384,length=5000):
        length=min(81783,length)
        length=max(2,length)
        ds = load_dataset("jpawan33/fkr30k-image-captioning-dataset")
        self.ds=ds["train"]
        self.image_size=image_size
        self.sd_image_size=sd_image_size
        self.data=[]
        self.sd_data=[]
        self.transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.CenterCrop(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
            ])
        self.sd_transform = transforms.Compose([
            transforms.Resize(self.sd_image_size),
            transforms.CenterCrop(self.sd_image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
            ])
        for idx in range(length):
            self.data.append(self.transform(self.ds[idx]["image"].convert("RGB")))
            self.sd_data.append(self.sd_transform(self.ds[idx]["image"].convert("RGB")))
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        # データ1とデータ2をセットで返す
        return self.data[idx],self.sd_data[idx]
    
class valDataset_single(Dataset):
    def __init__(self,image_size=224,start_index=5000,length=100):
        length=min(82783-start_index,length)
        length=max(2,length)
        ds = load_dataset("jpawan33/fkr30k-image-captioning-dataset")
        self.ds=ds["train"]
        self.image_size=image_size
        self.data=[]
        self.start_index = start_index
        self.transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.CenterCrop(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
            ])
        for idx in range(length):
            idx=idx+start_index
            self.data.append(self.transform(self.ds[idx]["image"].convert("RGB")))
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        # データ1とデータ2をセットで返す
        return self.data[idx]
    def get_samples(self):
        return self.data[0],self.data[1],self.data[2]
    def get_image_samples(self):
        return self.ds[0+self.start_index]["image"],self.ds[1+self.start_index]["image"],self.ds[2+self.start_index]["image"]