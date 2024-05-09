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
import time

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
    np1 = np.array(points1)
    np2 = np.array(points2)

    distances = np.linalg.norm(np1 - np2, axis=1)
    return np.sum(distances)



device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")
env = environment.FrankaArmEnvironment()

keypointExtracter = cv2.ORB_create()#10000, fastThreshold=0)
keypointMatcher = cv2.BFMatcher()#cv2.NORM_HAMMING, crossCheck=True)

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
kp_init, des_init = keypointExtracter.detectAndCompute(img_init_rgb, None)

plt.ion()
plt.show()

ERR_THRESHOLD = 50 #generic error between the two sets of points
error = ERR_THRESHOLD + 1
while error > ERR_THRESHOLD:
    time.sleep(5)
    #save live image to temp folder
    env.robotSaveCameraSnapshot("live", dir_path + "\\temp")

    with torch.no_grad(): #TODO: is this still needed?
        
        img_live_rgb = cv2.imread(path_live_rgb, cv2.IMREAD_GRAYSCALE)
        kp_live, des_live = keypointExtracter.detectAndCompute(img_live_rgb, None)
        # Brute force greedy match keypoints based on descriptors, as a first guess
        matches = keypointMatcher.match(des_live, des_init)
        matches = sorted(matches, key = lambda x:x.distance)
        matches = matches[:10]
        # GMS (Grid-based Motion Statistics) algorithm refines the guess for high quality matches
        #matches_gms = cv2.xfeatures2d.matchGMS(img_live_rgb.shape, img_init_rgb.shape, kp_live, kp_init, matches, withRotation=True, withScale=True)

        matchImg = cv2.drawMatches(img_live_rgb,kp_live,img_init_rgb,kp_init,matches,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        plt.imshow(matchImg)
        plt.pause(0.1)

        #Extract matching coordinates
        points_live, points_init = [], []
        for m in matches:
            x, y = kp_live[m.queryIdx].pt
            points_live.append( (int(x), int(y)) )

            u, v = kp_init[m.trainIdx].pt
            points_init.append( (int(u), int(v)) )

        if len(points_live) == 0 or len(points_init) == 0:
            raise Exception("No keypoints found. This is a big problem!")

        #Given the pixel coordinates of the correspondences, add the depth channel
        points_live = add_depth(points_live, path_live_depth)
        points_init = add_depth(points_init, path_init_depth)
        R, t = find_transformation(points_live, points_init)

        print(R)
        print(t)

        #A function to convert pixel distance into meters based on calibration of camera.
        t_meters = t * 0.001
        #I think its 1 pixel is 1mm according to some sources, move to a env method

        #Move robot
        env.robotMoveEefPosition(t_meters,R)
        error = compute_error(points_live, points_init)
        print(error)

plt.close()

#execute demo
with open(dir_path + f"\\demonstrations\\{best[1]}.pkl", 'rb') as f:
    trace = pickle.load(f)

#--- we need to add the alligned eef pos and orn to each keyframe of the trace
# but actually we only need to add the difference between the final alligned pos and the rest pos we started at.
# so we need to so like demo + aligned - rest which is fine for pos but how do you do that for orn???
alligned_pos, alligned_orn = env.robotGetEefPosition()

for keyFrame in range(len(trace)):
    demo_pos, demo_orn, demo_gripper = trace[keyFrame]

    offset_pos, offset_orn = env.offsetMovementInverse(alligned_pos, alligned_orn, env.restPos, env.restOrn)
    desired_pos, desired_orn = env.offsetMovement(demo_pos, demo_orn, offset_pos, offset_orn)

    print(offset_pos, offset_orn)
    print(desired_pos, desired_orn)

    env.robotSetEefPosition(desired_pos, desired_orn)
    env.robotCloseGripper() if demo_gripper else env.robotOpenGripper()
#---

while True:
    env.stepEnv()