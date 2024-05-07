import environment
import torch
import numpy as np 
import torchvision.transforms as T
from PIL import Image
import cv2
import pickle
import glob
import os
import matplotlib.pyplot as plt

dir_path = os.path.dirname(os.path.realpath(__file__))


dino = torch.hub.load('facebookresearch/dino:main', 'dino_vitb8')

image_transforms = T.Compose([
    T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])

def embedImage(path):
    print(path)
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
            embeddings[path.split("\\")[-1].split("-")[0]] = embedImage(path)
            #path name shouldnt have a - in it (apart from the one at the end) otherwise itll break. maybe fix that
            i += 1
        except:
            print(f"An error occured with image {path}")

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
    #TODO can i do this with cv2 and then not need PIL as a dependency???
    depthImg = Image.open(depth_path)
    depth = env.calculateDepthFromBuffer(np.array(depthImg))
    return [(x,y, depth[y,x]) for (x,y) in points]


def compute_error(points1, points2):
    print(points1[:5])
    print(points2[:5])
    #testing, delete this, the points sholdnt be identical but they are.
    # I think an artefact of the exact images, because the cup is kind of out of view, the keypoints
    # all care about the table, which hasnt moved so no error in keypoints
    
    np1 = np.array(points1)
    np2 = np.array(points2)

    distances = np.linalg.norm(np1 - np2, axis=1)
    return np.sum(distances)



device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")
env = environment.FrankaArmEnvironment()

orb = cv2.ORB_create(10000, fastThreshold=0)
matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

path_live_rgb = dir_path + f"\\temp\\live-rgb.jpg"
path_live_depth = dir_path + f"\\temp\\live-depth.jpg"

#take initial screenshot
env.robotSaveCameraSnapshot("live", dir_path + "\\temp")
init_emb = embedImage(path_live_rgb)

demo_img_emb = loadEmbeddings()
cos_sim = torch.nn.CosineSimilarity(dim=1, eps=1e-08)
best = (-1, None)
for key in demo_img_emb.keys():
    sim = cos_sim(init_emb, demo_img_emb[key])
    if sim > best[0]:
        best = (sim, key)

#load bottleneck image
path_init_rgb = dir_path + f"\\demonstrations\\{best[1]}-rgb.jpg"
path_init_depth = dir_path + f"\\demonstrations\\{best[1]}-depth.jpg"

img_init_rgb = cv2.imread(path_init_rgb, cv2.IMREAD_GRAYSCALE)
kp_init, des_init = orb.detectAndCompute(img_init_rgb, None)


ERR_THRESHOLD = 50 #generic error between the two sets of points
error = ERR_THRESHOLD + 1
while error > ERR_THRESHOLD:
    #save live image to temp folder
    env.robotSaveCameraSnapshot("live", dir_path + "\\temp")

    with torch.no_grad(): #TODO: is this still needed?
        
        img_live_rgb = cv2.imread(path_live_rgb, cv2.IMREAD_GRAYSCALE)
        kp_live, des_live = orb.detectAndCompute(img_live_rgb, None)
        # Brute force greedy match keypoints based on descriptors, as a first guess
        matches = matcher.match(des_live, des_init)
        # GMS (Grid-based Motion Statistics) algorithm refines the guess for high quality matches
        matches_gms = cv2.xfeatures2d.matchGMS(img_live_rgb.shape, img_init_rgb.shape, kp_live, kp_init, matches, withRotation=True, withScale=True)

        #Extract matching coordinates
        points_live, points_init = [], []
        for m in matches_gms:
            x, y = kp_live[m.queryIdx].pt
            points_live.append( (int(x), int(y)) )

            u, v = kp_init[m.trainIdx].pt
            points_init.append( (int(u), int(v)) )

        #Given the pixel coordinates of the correspondences, add the depth channel
        points_live = add_depth(points_live, path_live_depth)
        points_init = add_depth(points_init, path_init_depth)
        R, t = find_transformation(points_live, points_init)

        #A function to convert pixel distance into meters based on calibration of camera.
        t_meters = t * 0.001
        #I think its 1 pixel is 1mm according to some sources, move to a env method

        #Move robot
        env.robotMoveEefPosition(t_meters,R)
        error = compute_error(points_live, points_init)
        print(error)