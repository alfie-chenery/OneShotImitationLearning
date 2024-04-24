import environment
import torch
import numpy as np 
import torchvision.transforms as T
from PIL import Image
import pickle
import glob
import os
from dinofeatures.correspondences import find_correspondences, draw_correspondences
import matplotlib.pyplot as plt

dir_path = os.path.dirname(os.path.realpath(__file__))

#Hyperparameters for DINO correspondences extraction
num_pairs = 8
load_size = 224
layer = 9
facet = 'key'
bin=True
thresh=0.05
model_type='dino_vits8' #vitb8
stride=4
ERR_THRESHOLD = 50 #generic error between the two sets of points

#Download and load DINO
dino = torch.hub.load('facebookresearch/dino:main', model_type)

#Create an image transform pipeline
image_transforms = T.Compose([
    T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])


def embedImage(path):
    img = Image.open(path)
    img = image_transforms(img)
    img = img.unsqueeze(0)
    emb = dino(img)
    return emb


def loadEmbeddings():
    """
    Load demonstations context images and embed using dino
    """
    embeddings = {}

    i = 0
    for path in glob.glob(dir_path + "\\demonstrations\\*-rgb.jpg"):
        try:
            name = path.split("\\")[-1].split(".")[0]
            emb = embedImage(path)
            embeddings[name] = emb
            i += 1
        except:
            print("An error occured with one of the images")

    print(f"Successfully loaded {i} images")
    return embeddings


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


def add_depth(points, depth_path):
    #probably calculate actual depth from buffer here using env.calculateDepthFromBuffer()
    depthImg = Image.open(depth_path)
    depth = env.calculateDepthFromBuffer(np.array(depthImg))

    return [(y,x, depth[y,x]) for (y,x) in points]
    #also check that the points are of the form (y,x) it seems like it but test


def compute_error(points1, points2):
    np1 = np.array(points1)
    np2 = np.array(points2)
    return np.linalg.norm(np1 - np2)



#Robot arm environment
env = environment.FrankaArmEnvironment()

#Save current image, find most similar demo image
env.robotSaveCameraSnapshot("initial_scene", dir_path + "\\temp")
initial_emb = embedImage(dir_path + "\\temp\\initial_scene-rgb.jpg")

emb1 = dir_path + "\\temp\\mouse.jpg"
emb2 = dir_path + "\\temp\\rat.jpg"
print("before")
points1, points2, image1_pil, image2_pil = find_correspondences(emb1, emb2, num_pairs, load_size, layer, facet, bin, thresh, model_type, stride, dino)
print(points1, points2)
fig1, fig2 = draw_correspondences(points1, points2, image1_pil, image2_pil)
fig1.savefig(dir_path + "\\fig1.png", bbox_inches='tight', pad_inches=0)
fig2.savefig(dir_path + "\\fig2.png", bbox_inches='tight', pad_inches=0)
print("HELP!")

img_embeddings = loadEmbeddings()
cos_sim = torch.nn.CosineSimilarity(dim=1, eps=1e-08)
best = (-1, None)

for key in img_embeddings.keys():
    sim = cos_sim(initial_emb, img_embeddings[key])
    if sim > best[0]:
        best = (sim, key)

#load bottleneck image
rgb_bn_path = dir_path + f"\\demonstrations\\{best[1]}-rgb.jpg"
depth_bn_path = dir_path + f"\\demonstrations\\{best[1]}-depth.jpg"

error = 100000
while error > ERR_THRESHOLD:
    #save live image to temp folder
    env.robotSaveCameraSnapshot("live", dir_path + "\\temp")
    rgb_live_path = dir_path + f"\\temp\\live-rgb.jpg"
    depth_live_path = dir_path + f"\\temp\\live-depth.jpg"

    with torch.no_grad():
        points1, points2, image1_pil, image2_pil = find_correspondences(rgb_live_path, rgb_bn_path, num_pairs, load_size, layer,
                                                                            facet, bin, thresh, model_type, stride, dino)
        #Given the pixel coordinates of the correspondences, add the depth channel
        points1 = add_depth(points1, depth_bn_path)
        points2 = add_depth(points2, depth_live_path)
        R, t = find_transformation(points1, points2)

        #A function to convert pixel distance into meters based on calibration of camera.
        t_meters = t * 0.001
        #I think its 1 pixel is 1mm according to some sources, move to a env method

        #Move robot
        env.robotMoveEefPosition(t_meters,R)
        error = compute_error(points1, points2) #probs just use euclidian distance?