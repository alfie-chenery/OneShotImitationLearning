#Follow the steps to install DINO here: https://github.com/facebookresearch/dino.


import torch
import numpy as np 
import matplotlib.pyplot as plt 
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms,utils
from torchvision.datasets import ImageFolder
import warnings 
import glob
import time
import os
#warnings.filterwarnings("ignore")


from PIL import Image
import torchvision.transforms as T
# import hubconf


#Download and load DINO
dino = torch.hub.load('facebookresearch/dino:main', 'dino_vitb8')
#dino = dino.cuda().float()


#Create an image transform pipeline
image_transforms = T.Compose([
    T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])


#Embed all images in your folder and create a dictionary of names as keys and embs as values
dir_path = os.path.dirname(os.path.realpath(__file__))
print(dir_path)
print(dir_path + "\content\*.jpg")

im_embs = {}
for path in glob.glob(dir_path + "\content\*.jpg"):
    img = Image.open(path)
    start = time.time()
    img = image_transforms(img)
    img = img.unsqueeze(0)
    #img = img.cuda().float()
    emb = dino(img)
    print(time.time() - start)
    im_embs[path.split("\\")[-1].split(".")[0]] = emb

print(im_embs.keys())


#Create a cosine similarity object
cos_sim = torch.nn.CosineSimilarity(dim=1, eps=1e-08)


#Compute cosine similarities
for a in ["mouse", "bottle"]:
    for b in ["wallet", "rat"]:
        print(a,b)
        print(cos_sim(im_embs[a], im_embs[b]))