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
    return [(x,y, depth[int(y),int(x)]) for (x,y) in points]
    #TODO consider interpolating the pixels around (y,x) rather than just rounding to nearest int


def compute_error(points1, points2):    
    np1 = np.array(points1)
    np2 = np.array(points2)

    distances = np.linalg.norm(np1 - np2, axis=1)
    return np.mean(distances)
    #We cant be certain how many keypoint matches we get from the GMS matching. Therefore,
    # to be fair we should normalise by the number of matches. Ie mean instead of sum



device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")

env = environment.FrankaArmEnvironment()
for _ in range(250):
    env.stepEnv()
    #Let the environment come to rest before starting


keypointExtracter = cv2.ORB_create(10000, fastThreshold=0)
keypointMatcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

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

ERR_THRESHOLD = 1 #generic error between the two sets of points
error = ERR_THRESHOLD - 1 #TESTING TO SKIP THIS ALIGNMENT, MAKE + 1 TO ACTUALLY WORK     <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
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
        # l = len(matches)
        # print(l)
        matches = matches[:500]
        # GMS (Grid-based Motion Statistics) algorithm refines the guess for high quality matches
        matches_gms = cv2.xfeatures2d.matchGMS(img_live_rgb.shape, img_init_rgb.shape, kp_live, kp_init, matches, withRotation=True, withScale=True)

        matchImg = cv2.drawMatches(img_live_rgb,kp_live,img_init_rgb,kp_init,matches,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        plt.imshow(matchImg)
        plt.pause(0.1)

        #Extract matching coordinates
        points_live, points_init = [], []
        for m in matches:
            x, y = kp_live[m.queryIdx].pt
            points_live.append( (x, y) )

            u, v = kp_init[m.trainIdx].pt
            points_init.append( (u, v) )

        if len(points_live) == 0 or len(points_init) == 0:
            #raise Exception("No keypoints found. This is a big problem!")
            env.robotSetJointAngles(env.restPoses)
            #env.resetEnv()
            continue
            #move the robot back to start position and try again

        #Given the pixel coordinates of the correspondences, add the depth channel
        points_live = add_depth(points_live, path_live_depth)
        points_init = add_depth(points_init, path_init_depth)
        R, t = find_transformation(points_live, points_init)

        #Convert pixel distance into meters based on calibration of camera.
        t_meters = env.pixelsToMetres(t)
        print(t_meters)

        #Move robot
        env.robotMoveEefPosition(t_meters,R)

        error = compute_error(points_live, points_init)
        print(error)

plt.close()

#execute demo
with open(dir_path + f"\\demonstrations\\{best[1]}.pkl", 'rb') as f:
    trace = pickle.load(f)

alligned_pos, alligned_orn = env.robotGetEefPosition()

#testing
alligned_pos = [p+t for (p,t) in zip(env.restPos, [0,0.05,0])]
alligned_orn = env.getQuaternionFromMatrix(np.dot(env.getMatrixFromQuaternion(env.restOrn),env.getMatrixFromQuaternion(env.getQuaternionFromEuler([0,0,np.pi/3]))))
#---- should print out offset 0.05 and pi/3

offset_pos, offset_orn = env.calculateOffset(env.restPos, env.restOrn, alligned_pos, alligned_orn)
print(f"Alligned offset: {offset_pos}, {offset_orn}")
print(env.getEulerFromQuaternion(offset_orn))

#---testing
# offset_pos = [0,0.05,0]
# offset_orn = env.getQuaternionFromEuler([0,0,np.pi/3])
#--- Should pick up brick by the sides not corners

offset_mat = env.getMatrixFromQuaternion(offset_orn)

for keyFrame in range(len(trace)):
    demo_pos, demo_orn, demo_gripper = trace[keyFrame]
    desired_pos, desired_orn = env.offsetMovementLocal(demo_pos, demo_orn, offset_pos, offset_mat)

    env.robotSetEefPosition(desired_pos, desired_orn, interpolationSteps=250)
    env.robotCloseGripper() if demo_gripper else env.robotOpenGripper()


while True:
    env.stepEnv()


"""
I think the reson its fucked is the eef is kinda upside down. So if we need to move it 0.2 in the x direction (world coords)
then we actually need to move -0.2 in local eef coords. Im not sure how this affects the rotation.
Just play about until both tests work as expected and the arm is moving in the direction youd expect

The calculateOffset function is working. Dont change this. It calculates the offset that occured correctly
Whats going wrong is what we do with that offset. The offsetMovementLocal needs changing to account for the fact
the eef is rotated 180
"""