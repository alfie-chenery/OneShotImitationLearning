import environment
import torch
import numpy as np 
import torchvision.transforms as T
from PIL import Image
import pickle
import glob
import os
from dinofeatures.correspondences import find_correspondences


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


def loadDemonstrations():
    """
    Load demonstations and context images. Automatically emed using image transformer
    """
    demonstrations = {}
    dir_path = os.path.dirname(os.path.realpath(__file__))

    i = 0
    for path in glob.glob(dir_path + "\\demonstrations\\*.pkl"):
        try:
            name = path.split("\\")[-1].split(".")[0]

            with open(path, 'rb') as f:
                trace = pickle.load(f)

            rgb = Image.open(dir_path + f"\\{name}-rgb.jpg")
            rgb = image_transforms(rgb)
            rgb = rgb.unsqueeze(0)
            emb = dino(rgb)

            depth = Image.open(dir_path + f"\\{name}-depth.jpg")

            #TODO do i want to store the rgb and in what form? we defo need depth to add it back in for find correspondences, but is rgb necessary or do i just need embedding?
            # actually what do i even want to store here. Find correspodences weirdly opens PIl images itself from filenames, so maybe some of this is better openned on the fly
            # yeah probably best not to do that here. we select the single demonstration to follow from embeding. Then we can only open the rgb and depth file for the ONE demo we use, not all of them

            demonstrations[name] = (emb, trace, rgb, depth)
            i += 1
        except:
            print("An error occured with one of the demonstrations")

    print(f"Successfully loaded {i} demonstrations")
    return demonstrations


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
    #probably calculate actual depth from buffer here using env.calculateDepthFromBuffer()
    return [(y,x, depth[y,x]) for (y,x) in points]
    #also check that the points are of the form (y,x) it seems like it but test


def compute_error(points1, points2):
    np1 = np.array(points1)
    np2 = np.array(points2)
    return np.linalg.norm(np1 - np2)



#Create a cosine similarity object
cos_sim = torch.nn.CosineSimilarity(dim=1, eps=1e-08)

demos = loadDemonstrations()

#Robot arm environment
env = environment.FrankaArmEnvironment()

_, _, rgb_bn, depth_bn, _ = env.robotGetCameraSnapshot() 
#take picture, find the demo which matches the image the best
#select bottleneck image

# #Compute cosine similarities
# for a in ["mouse", "bottle"]:
#     for b in ["wallet", "rat"]:
#         print(a,b)
#         print(cos_sim(imageEmbeddings[a], imageEmbeddings[b]))


error = 100000
ERR_THRESHOLD = 50 #A generic error between the two sets of points

while error > ERR_THRESHOLD:
    _, _, rgb_live, depth_live, _ = env.robotGetCameraSnapshot() #save live image into temp
    with torch.no_grad():
        points1, points2, image1_pil, image2_pil = find_correspondences(rgb_live, rgb_bn, num_pairs, load_size, layer,
                                                                            facet, bin, thresh, model_type, stride)
        #Given the pixel coordinates of the correspondences, add the depth channel
        points1 = add_depth(points1, depth_bn)
        points2 = add_depth(points2, depth_live)
        R, t = find_transformation(points1, points2)

        #A function to convert pixel distance into meters based on calibration of camera.
        t_meters = t * 0.001
        #I think its 1 pixel is 1mm according to some sources

        #Move robot
        env.robotMoveEefPosition(t_meters,R)
        error = compute_error(points1, points2) #probs just use euclidian distance?