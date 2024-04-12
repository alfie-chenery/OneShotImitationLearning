import environment
import torch
import numpy as np 
import torchvision.transforms as T
from PIL import Image
import glob
import os


#Hyperparameters for DINO correspondences extraction
num_pairs = 8
load_size = 224
layer = 9
facet = 'key'
bin=True
thresh=0.05
model_type='dino_vitb8'
stride=4


#Download and load DINO
dino = torch.hub.load('facebookresearch/dino:main', 'dino_vitb8')

#Create an image transform pipeline
image_transforms = T.Compose([
    T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])

imageEmbeddings = {}


def loadImages():
    #Load saved images and embed using transformer
    dir_path = os.path.dirname(os.path.realpath(__file__))

    i = 0
    for path in glob.glob(dir_path + "\\image_snapshots\\*.jpg"):
        img = Image.open(path)
        img = image_transforms(img)
        img = img.unsqueeze(0)
        emb = dino(img)

        imageEmbeddings[path.split("\\")[-1].split(".")[0]] = emb
        i += 1
    
    print(f"Successfully loaded {i} images")


def find_transformation(X, Y):
    #Find transformation given two sets of correspondences between 3D points
    # Calculate centroids
    cX = np.mean(X, axis=0)
    cY = np.mean(Y, axis=0)
    # Subtract centroids to obtain centered sets of points
    Xc = X - cX
    Yc = Y - cY
    # Calculate covariance matrix
    C = np.dot(Xc.T, Yc)
    # Compute SVD
    U, S, Vt = np.linalg.svd(C)
    # Determine rotation matrix
    R = np.dot(Vt.T, U.T)
    # Determine translation vector
    t = cY - np.dot(R, cX)
    return R, t


def add_depth(points, depth):
    return [(y,x, depth[y,x]) for (y,x) in points]



#Create a cosine similarity object
cos_sim = torch.nn.CosineSimilarity(dim=1, eps=1e-08)

loadImages()

#Compute cosine similarities
for a in ["mouse", "bottle"]:
    for b in ["wallet", "rat"]:
        print(a,b)
        print(cos_sim(imageEmbeddings[a], imageEmbeddings[b]))

#Robot arm environment
env = environment.FrankaArmEnvironment()

env.robotGetCameraSnapshot()