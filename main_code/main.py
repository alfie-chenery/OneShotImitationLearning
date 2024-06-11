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
import keyboard

#Custom exception type
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


def loadDemo(name):
    rgbPath = dir_path + f"\\demonstrations\\{name}-rgb.jpg"
    depthPath = dir_path + f"\\demonstrations\\{name}-depth.pkl"
    vmPath = dir_path + f"\\demonstrations\\{name}-vm.pkl"
    tracePath = dir_path + f"\\demonstrations\\{name}.pkl"

    rgb = cv2.imread(rgbPath, cv2.IMREAD_GRAYSCALE)
    with open(depthPath, 'rb') as f:
        depth = pickle.load(f)
    with open(vmPath, 'rb') as f:
        vm = pickle.load(f)
    with open(tracePath, 'rb') as f:
        trace = pickle.load(f) 

    return (rgb, depth, vm, trace)


def findTransformation(P, Q):
    """
    Find transformation given two sets of correspondences between 3D points
    The transformation maps Q onto P (as close as possible)
    """
    assert P.shape == Q.shape
    n, m = P.shape

    centroidP = np.mean(P, axis=0)
    centroidQ = np.mean(Q, axis=0)

    #Subtract centroids to obtain centered sets of points
    Pcentered = P - centroidP
    Qcentered = Q - centroidQ

    cov = (Pcentered.T @ Qcentered) / n

    #Compute Single Value Decomposition
    U, E, Vt = np.linalg.svd(cov)

    #Create sign matrix to correct for reflections
    sign = np.linalg.det(U) * np.linalg.det(Vt)
    assert abs(sign) > 0.99  #should be very close to 1 or -1 (floating point errors)
    sign = round(sign)       #fix said floating point rounding errors
    S = np.diag([1] * (m-1) + [sign])

    #Compute rotation matrix
    R = U @ S @ Vt
    
    t = centroidP - centroidQ

    #Qprime = np.array([t + R @ q for q in Q])

    return R, t


def convertToWorldCoords(points, depth_map, viewMatrix):
    """
    Takes in a list of (x,y) keypoints as pixel locations found from the image, and depth array
    Returns numpy array of shape (N,3). Rows of (X,Y,Z) keypoints in world coordinates
    """
    depth = depth_map
    h, w = depth.shape
    out = []

    projectionMatrix = np.array(env.projectionMatrix).reshape((4,4), order='F')
    viewMatrix = np.array(viewMatrix).reshape((4,4), order='F')
    pixel2World = np.linalg.inv(np.matmul(projectionMatrix, viewMatrix))

    #TODO: comment this. Its like converting to NDC pos. Should write about all this in the report

    for (x,y) in points:
        X = (2*x - w)/w
        Y = -(2*y - h)/h
        Z = 2*depth[int(y), int(x)] - 1
        pixPos = np.array([X, Y, Z, 1], dtype=np.float64)          #homogoneous coordinates
        position = np.matmul(pixel2World, pixPos)
        position = position / position[3]

        out.append(position.tolist()[:3])

    return np.array(out, dtype=np.float64)  

    


def computeError(points1, points2):    
    distances = np.linalg.norm(points1 - points2, axis=1)
    return np.mean(distances)
    #We cant be certain how many keypoint matches we get from the GMS matching. Therefore,
    # to be fair we should normalise by the number of matches. Ie mean instead of sum


def extractCorrespondingKeypoints(img_live, img_init, displayMatches=True):
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
    matchesGMS = cv2.xfeatures2d.matchGMS(img_live.shape[:2], img_init.shape[:2], kp_live, kp_init, matches, withRotation=True, withScale=False)
    #matchesGMS = matches

    matchesGMS = sorted(matchesGMS, key=lambda x:x.distance)

    # n = len(matchesGMS)
    # matchesGMS = matchesGMS[:n//2] # take only the best half of matches
    #matchesGMS = matchesGMS[:500]

    if len(matchesGMS) == 0:
        raise NoKeypointsException("Could not match any keypoints")

    if displayMatches:
        matchImg = cv2.drawMatches(img_live, kp_live, img_init, kp_init, matchesGMS, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        plt.figure(figsize = (8,6))
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

env = environment.FrankaArmEnvironment(videoLogging=False, out_dir=dir_path+"\\out")

keypointExtracter = cv2.ORB_create(10000, fastThreshold=0)
#keypointExtracter = cv2.SIFT_create()
keypointMatcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
#keypointMatcher = cv2.BFMatcher()

drawKeypointsInWorld = True


plt.ion()

#take initial screenshot
env.robotSaveCameraSnapshot("init", dir_path + "\\temp")
init_emb = embedImage(dir_path + "\\temp\\init-rgb.jpg")

demo_img_emb = loadEmbeddings()
cos_sim = torch.nn.CosineSimilarity(dim=1, eps=1e-08)
best = (-1, None)
for key in demo_img_emb.keys():
    sim = cos_sim(init_emb, demo_img_emb[key])
    if sim > best[0]:
        best = (sim, key)

#load bottleneck image
print(f"Best match found: {best[1]}")
demo_rgb, demo_depth, demo_vm, demo_trace = loadDemo(best[1])

offset_pos = np.zeros(3)
offset_ornMat = np.eye(3)
iter = 0
ERR_THRESHOLD = 0.2 #average error between sets of points (meters)
error = ERR_THRESHOLD + 1
while error > ERR_THRESHOLD: # or iter < 2:

    _, _, live_rgb, live_depth, _, live_vm = env.robotGetCameraSnapshot()
    
    try:
        points_live, points_demo = extractCorrespondingKeypoints(live_rgb, demo_rgb, displayMatches=True)

        #TESTING
        # points_demo = [(10, 445),(140, 445),(10, 574),(140, 574)]
        # points_live = [(421, 650),(551, 650),(421, 779),(551, 779)] #0.5, 0.05 unrotated
        #points_live = [(401, 738),(462, 629), (508, 800), (570, 693)] #0.5, 0.05, rotated pi/3

        points_live = convertToWorldCoords(points_live, live_depth, live_vm)
        points_demo = convertToWorldCoords(points_demo, demo_depth, demo_vm)

        if drawKeypointsInWorld:
            #limit the amount we draw to ~150, after this it gets too slow.
            # dont just draw the first 150 as they will be all over one side.
            step = int(np.ceil((len(points_live) / 150)))

            for i in range(0, len(points_live), step):
                pos = points_live[i]
                env.addDebugLine(pos, (0,0,1), [1,0,1], True)
            for i in range(0, len(points_demo), step):
                pos = points_demo[i]
                env.addDebugLine(pos, (0,0,1), [0,1,1], False)
            env.drawDebugLines()

        # print(points_live)
        # print(points_demo)

        R, t = findTransformation(points_live, points_demo)
        print(f"Incremental Update:\n  Translation:{t},\n  Rotation (Matrix):\n{R}\n  Rotation (euler):{env.getEulerFromMatrix(R)}\n")

        error = computeError(points_live, points_demo)
        print(f"Error: {error}")

        # After finding keypoints, wait so we can check the correspondence image
        print("Press Space to continue...")
        keyboard.wait("space")

        env.robotMoveEefPosition(t,R)
        
    except NoKeypointsException as e:
        print(e)
        env.robotSetJointAngles(env.restPoses)
        randomTranslation = np.append(np.random.uniform(-0.2, 0.2, 2), 0).tolist()
        env.robotMoveEefPosition(randomTranslation, np.identity(3))
        offset_pos = np.zeros(3)
        offset_ornMat = np.eye(3)
        continue
    
    # Cumulative offset so far
    offset_pos = t + offset_pos
    offset_ornMat = R @ offset_ornMat
    iter += 1
    
    #TESTING
    # break


plt.close()
env.removeAllDebugLines()

for i in range(50):
    env.stepEnv()

#We are alligned, calculate the offset to apply to demo keyframes
# alligned_pos, alligned_orn = env.robotGetEefPosition()
# offset_pos, offset_orn = env.calculateOffset(env.restPos, env.restOrn, alligned_pos, alligned_orn)
# offset_ornMat = env.getMatrixFromQuaternion(offset_orn)
#Above method seems to have some inacuracies. Likely due to dynamics errors slowly accumulating
#Alternative method, offset is tracked throughout and built up as the cumulative of each incremental update

print(f"Alligned offset:\n  Translation:{offset_pos},\n  Rotation (matrix):\n{offset_ornMat}\n Rotation (euler):{env.getEulerFromMatrix(offset_ornMat)}\n")





#Execute demo with offset applied
for keyFrame in range(len(demo_trace)):
    demo_pos, demo_orn, demo_gripper = demo_trace[keyFrame]
    desired_pos, desired_orn = env.offsetMovementLocal(demo_pos, demo_orn, offset_pos, offset_ornMat)

    env.robotSetEefPosition(desired_pos, desired_orn, interpolationSteps=250)
    env.robotCloseGripper() if demo_gripper else env.robotOpenGripper()



for _ in range(100):
    env.stepEnv()

# env.closeEnv()

