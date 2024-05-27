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
import keyboard
from scipy.spatial.transform import Rotation

class NoKeypointsException(Exception):
    pass

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


def find_transformation(P, Q):
    """
    Find transformation given two sets of correspondences between 3D points
    The transformation maps P onto Q (as close as possible)
    """
    assert P.shape == Q.shape
    n, m = P.shape

    centroidP = np.mean(P, axis=0)
    centroidQ = np.mean(Q, axis=0)
    varianceQ = np.var(Q, axis=0)

    #Subtract centroids to obtain centered sets of points
    Pcentered = P - centroidP
    Qcentered = Q - centroidQ

    cov = (Pcentered.T @ Qcentered) / n

    #Compute Single Value Decomposition
    U, E, Vt = np.linalg.svd(cov)

    #Create sign matrix to correct for reflections
    sign = np.linalg.det(U) * np.linalg.det(Vt)
    S = np.diag([1] * (m-1) + [sign])

    #Compute rotation matrix
    #R = Vt.T @ S @ U
    R = U @ S @ Vt
    
    #if the rotation is the wrong way just transpose it lol

    #Compute scale factor
    c = varianceQ / np.trace(np.diag(E) @ S) #maybe need to use var P instead?
    #print(c)

    #Compute translation vector
    #t = centroidQ - np.dot(R, centroidP)
    #t = centroidP - c * R @ centroidQ
    t = centroidQ - R @ centroidP

    # t[0] -= 0.15
    # t[1] += 0.05
    # t[2] = -0.05

    #P_prime = np.array([t + c * R @ q for q in Q])

    return R, t


def convert_to_world_coords(points, depth_path, viewMatrix):
    """
    Takes in a list of (x,y) keypoints as pixel locations found from the image, and path to the depth image
    Returns numpy array of shape (N,3). Rows of (X,Y,Z) keypoints in world coordinates
    """
    #TODO can i do this with cv2 and then not need PIL as a dependency???
    depthImg = Image.open(depth_path)
    depth = env.calculateDepthFromBuffer(np.array(depthImg))
    h, w = depth.shape
    out = []

    projectionMatrix = np.array(env.projectionMatrix).reshape((4,4), order='F')
    viewMatrix = np.array(viewMatrix).reshape((4,4), order='F')
    pixel2World = np.linalg.inv(projectionMatrix @ viewMatrix)

    #TODO: comment this. Its like converting to NDC pos. Should write about all this in the report

    for (x,y) in points:
        X = (2*x - w)/w
        Y = -(2*y - h)/h
        Z = 2*depth[y,x] - 1
        pixPos = np.array([X, Y, Z, 1], dtype=np.float64)          #homogoneous coordinates
        position = pixel2World @ pixPos
        position = position / position[3]

        out.append(position.tolist()[:3])

    return np.array(out, dtype=np.float64)  

    


def compute_error(points1, points2):    
    distances = np.linalg.norm(points1 - points2, axis=1)
    return np.mean(distances)
    #We cant be certain how many keypoint matches we get from the GMS matching. Therefore,
    # to be fair we should normalise by the number of matches. Ie mean instead of sum


def extract_corresponding_keypoints(img_live, img_init, displayMatches=True):
    kp_live, des_live = keypointExtracter.detectAndCompute(img_live, None)
    kp_init, des_init = keypointExtracter.detectAndCompute(img_init, None)

    if len(kp_live) == 0:
        raise NoKeypointsException("Could not find keypoints in live image")
    if len(kp_init) == 0 :
        raise NoKeypointsException("Could not find keypoints in init image")

    # Brute force greedy match keypoints based on descriptors, as a first guess
    matches = keypointMatcher.match(des_live, des_init)

    #TODO remove any matches which match between different objects using segmentation map

    # GMS (Grid-based Motion Statistics) algorithm refines the guesses for high quality matches
    #matchesGMS = cv2.xfeatures2d.matchGMS(img_live.shape[:2], img_init.shape[:2], kp_live, kp_init, matches, withRotation=True, withScale=True)
    matchesGMS = matches

    matchesGMS = sorted(matchesGMS, key=lambda x:x.distance)
    matchesGMS = matchesGMS[:50]

    if displayMatches:
        matchImg = cv2.drawMatches(img_live, kp_live, img_init, kp_init, matchesGMS, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        plt.imshow(matchImg)
        plt.pause(0.01)

    #Extract matching coordinates
    points_live, points_init = [], []
    for m in matchesGMS:
        x, y = kp_live[m.queryIdx].pt
        points_live.append( (x, y) )

        u, v = kp_init[m.trainIdx].pt
        points_init.append( (u, v) )

    return (points_live, points_init)



device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")

env = environment.FrankaArmEnvironment(videoLogging=True, out_dir=dir_path+"\\out")

keypointExtracter = cv2.ORB_create(10000, fastThreshold=0)
#keypointExtracter = cv2.SIFT_create()
keypointMatcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
#keypointMatcher = cv2.BFMatcher()



plt.ion()

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
print(f"Best match found: {best[1]}")
path_init_rgb = dir_path + f"\\demonstrations\\{best[1]}-rgb.jpg"
path_init_depth = dir_path + f"\\demonstrations\\{best[1]}-depth.jpg"

img_init_rgb = cv2.imread(path_init_rgb, cv2.IMREAD_GRAYSCALE)
initView = env.robotGetCameraViewMatrix()

ERR_THRESHOLD = 0.001 #generic error between the two sets of points
error = ERR_THRESHOLD + 1 #TESTING -1 TO SKIP THIS ALIGNMENT, MAKE + 1 TO ACTUALLY WORK     <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
while error > ERR_THRESHOLD:
    #save live image to temp folder
    env.robotSaveCameraSnapshot("live", dir_path + "\\temp")
        
    img_live_rgb = cv2.imread(path_live_rgb, cv2.IMREAD_GRAYSCALE)
    
    try:
        points_live, points_init = extract_corresponding_keypoints(img_live_rgb, img_init_rgb)

        #TESTING
        # points_init = [(10, 445),(140, 445),(10, 574),(140, 574)]
        # points_live = [(421, 650),(551, 650),(421, 779),(551, 779)]
        points_init = [(11, 446),(139, 446),(10, 573),(139, 573)]
        points_live = [(422, 651),(550, 651),(422, 778),(550, 778)]

        #TODO: the view matrix needs to be the one when the picture was taken, this is a new thing we need to save.
        # Should not be the current one

        points_live = convert_to_world_coords(points_live, path_live_depth, env.robotGetCameraViewMatrix())
        points_init = convert_to_world_coords(points_init, path_init_depth, initView)
        print(points_live)
        print(points_init)

        #the points should have the same z lets be real, making it so makes it much better 
        # ditch saving images and just pass np arrays around. Hopefully that improves z accuracy
        z = points_live[0,2]
        points_live[:, 2] = z
        print(points_live)
        z = points_init[0,2]
        points_init[:, 2] = z
        print(points_init)

        R, t = find_transformation(points_live, points_init)

        #TESTING
        #R = np.identity(3)

        print(f"Incremental Update:\n  Translation:{t},\n  Rotation (Matrix):\n{R}\n  Rotation (euler):{env.getEulerFromMatrix(R)}\n")

        error = compute_error(points_live, points_init)
        print(f"Error: {error}")

        # After finding keypoints, wait so we can check the correspondence image
        print("Press Space to continue...")
        keyboard.wait("space")

        env.robotMoveEefPosition(t,R)
        
    except NoKeypointsException as e:
        print(e)
        env.robotSetJointAngles(env.restPoses)
        # randomTranslation = np.append(np.random.uniform(-0.2, 0.2, 2), 0).tolist()
        # env.robotMoveEefPosition(randomTranslation, np.identity(3))
        continue

    #TESTING
    break


plt.close()

#We are alligned, calculate the offset to apply to demo keyframes
alligned_pos, alligned_orn = env.robotGetEefPosition()

offset_pos, offset_orn = env.calculateOffset(env.restPos, env.restOrn, alligned_pos, alligned_orn)
print(f"Alligned offset:\n  Translation:{offset_pos},\n  Rotation (quat):{offset_orn},\n  Rotation (euler):{env.getEulerFromQuaternion(offset_orn)}\n")

offset_ornMat = env.getMatrixFromQuaternion(offset_orn)

#execute demo
with open(dir_path + f"\\demonstrations\\{best[1]}.pkl", 'rb') as f:
    trace = pickle.load(f) 

for keyFrame in range(len(trace)):
    demo_pos, demo_orn, demo_gripper = trace[keyFrame]
    desired_pos, desired_orn = env.offsetMovementLocal(demo_pos, demo_orn, offset_pos, offset_ornMat)

    env.robotSetEefPosition(desired_pos, desired_orn, interpolationSteps=250)
    env.robotCloseGripper() if demo_gripper else env.robotOpenGripper()


while not keyboard.is_pressed("esc"):
    env.stepEnv()

