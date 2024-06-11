# File to test robustness of findTransformation when it recieves ideal keypoint matches with increasing amount of noise


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

dir_path = os.path.dirname(os.path.realpath(__file__))

def loadDemo(name):
    parent_dir = os.path.dirname(dir_path)
    rgbPath = parent_dir + f"\\demonstrations\\demo{name}-rgb.jpg"
    depthPath = parent_dir + f"\\demonstrations\\demo{name}-depth.pkl"
    vmPath = parent_dir + f"\\demonstrations\\demo{name}-vm.pkl"
    tracePath = parent_dir + f"\\demonstrations\\demo{name}.pkl"

    rgb = cv2.imread(rgbPath, cv2.IMREAD_GRAYSCALE)
    with open(depthPath, 'rb') as f:
        depth = pickle.load(f)
    with open(vmPath, 'rb') as f:
        vm = pickle.load(f)
    with open(tracePath, 'rb') as f:
        trace = pickle.load(f) 

    return (rgb, depth, vm, trace)


def loadLive(name):
    rgbPath = dir_path + f"\\live_data\\live{name}-rgb.jpg"
    depthPath = dir_path + f"\\live_data\\live{name}-depth.pkl"
    vmPath = dir_path + f"\\live_data\\live{name}-vm.pkl"

    rgb = cv2.imread(rgbPath, cv2.IMREAD_GRAYSCALE)
    with open(depthPath, 'rb') as f:
        depth = pickle.load(f)
    with open(vmPath, 'rb') as f:
        vm = pickle.load(f)

    return (rgb, depth, vm)


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


def actualTransform(item):
    if item == "Lego":
        t = [-0.1, 0.05, 0]
        E = [0, 0, np.pi/3]
        
    if item == "Mug":
        t = [0, 0.05, 0]
        E = [0, 0, np.pi/6]

    R = env.getMatrixFromEuler(E)
    return (np.array(t), R, np.array(E))


def extractIdealKeypoints(item):
    if item == "Lego":
        points_demo = [(10, 445),(140, 445),(10, 574),(140, 574)]     #[0.6, 0, 0.45] [0,0,0]
        points_live = [(401, 738),(462, 629), (508, 800), (570, 693)] #[0.5, 0.05, 0.45] [0,0,pi/3]
        #points_live = [(573,692), (509,803), (462,627), (397,739)] 

    if item == "Mug":
        points_demo = [(447,456), (452,662), (504,663), (505,455), (657,372), (699,317), (726,256), (735,186), (726,118), (701,56), (246,45), (216,112), (207,185), (216,255), (244,320), (284, 372)]     #[0.5, 0.05, 0.45] [0,0,0]
        points_live = [(315,723), (224,858), (291,885), (370,753), (538,765), (603,738), (659,694), (701,638), (726,578), (734,510), (339,281), (287,320), (242,378), (215,442), (207,509), (215,577)]    #[0.5, 0, 0.45] [0,0,pi/6]
    

    return (points_live, points_demo)


def addKeypointNoise(points, low, high):
    """
    Add 2d noise to the pixel coordinate of keypoints. Does not affect the depth image
    (but may now read the wrong pixel's depth). The whole point is to test poor keypoint matching
    Points is python list of (x,y) pairs
    """
    if low == 0 and high == 0:
        return points
    
    for i in range(len(points)):
        theta = np.random.uniform(0, 2 * np.pi)
        x = np.cos(theta)
        y = np.sin(theta)
        randomVector = np.array([x,y])
        randomMagnitude = np.random.uniform(low, high)
        assert np.isclose(np.linalg.norm(randomVector), 1.0) #should be unit vector

        randomVector *= randomMagnitude
        points[i] = (points[i][0] + randomVector[0], points[i][1] + randomVector[1])

    return points


def addCoordinateNoise(points, low, high):
    """
    Add 3d noise to world coordinates of keypoints
    Points is numpy array from convertToWorldCoords
    """
    if low == 0 and high == 0:
        return points
    
    for i in range(len(points)):
        theta = np.random.uniform(0, 2 * np.pi)
        z = np.random.uniform(-1, 1)
        x = np.sqrt(1 - np.square(z)) * np.cos(theta)
        y = np.sqrt(1 - np.square(z)) * np.sin(theta)
        randomVector = np.array([x,y,z])
        randomMagnitude = np.random.uniform(low, high)

        assert np.isclose(np.linalg.norm(randomVector), 1.0) #should be unit vector
        points[i] += (randomMagnitude * randomVector)

    return points


def distance_error(predicted, truth):
    return np.linalg.norm(predicted - truth)


def vector_error(predicted, truth):
    return predicted - truth


def rotation_error(predicted, truth):
    rel_R = np.dot(predicted.T, truth) # Compute the relative rotation matrix
    theta = np.arccos((np.trace(rel_R) - 1) / 2) # Compute the angle using the trace of the relative rotation matrix
    return theta



# SELECT WHICH TEST TO RUN
noise_mode = "coordinate_noise"  #["coordinate_noise", "pixel_noise"]



device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")

env = environment.FrankaArmEnvironment(videoLogging=False, out_dir=dir_path+"\\out")
env.robotGetCameraSnapshot()


with open(dir_path + f"\\results-{noise_mode}.txt", 'w') as f:
    for item in ["Lego", "Mug"]: #, "Ball", "Jenga", "Dominoes"]:
        print(item)

        actual_t, actual_R, actual_E = actualTransform(item)

        #load images
        demo_rgb, demo_depth, demo_vm, demo_trace = loadDemo(item)
        live_rgb, live_depth, live_vm = loadLive(item)

        print(f"ITEM: {item}", file=f)
        print(f"actual_t: {actual_t}", file=f)
        print(f"actual_E: {actual_E}", file=f)
        print(f"actual_R: {actual_R}\n", file=f)

        for low, high in [(0,0),(0,0.001),(0,0.01),(0,0.1)]:
            ts = []
            Rs = []
            Es = [] #euler angles of R
            for i in range(1 if (low == 0 and high == 0) else 1000):
                points_live, points_demo = extractIdealKeypoints(item)

                if noise_mode == "pixel_noise":
                    points_live = addKeypointNoise(points_live, low, high) #pixels

                points_live = convertToWorldCoords(points_live, live_depth, live_vm)
                points_demo = convertToWorldCoords(points_demo, demo_depth, demo_vm)
                
                if noise_mode == "coordinate_noise":
                    points_live = addCoordinateNoise(points_live, low, high) #meters

                R, t = findTransformation(points_live, points_demo)
                ts.append(t)
                Rs.append(R)
                Es.append(env.getEulerFromMatrix(R))

            errors = [vector_error(t,actual_t) for t in ts]
            avg_t_err = np.mean(errors, axis=0)

            errors = [distance_error(t,actual_t) for t in ts]
            avg_t_dist = np.mean(errors, axis=0)

            errors = [rotation_error(R,actual_R) for R in Rs]
            avg_R_err = np.mean(errors)

            errors = [vector_error(E,actual_E) for E in Es]
            avg_E_err = np.mean(errors, axis=0)

            errors = [distance_error(E,actual_E) for E in Es]
            avg_E_dist = np.mean(errors, axis=0)

            print(f"low={low}, high={high}:", file=f)
            print(f"avg t error: {avg_t_err}", file=f)
            print(f"avg t distance: {avg_t_dist}", file=f)
            print(f"avg E error: {avg_E_err}", file=f)
            print(f"avg E distance: {avg_E_dist}", file=f)
            print(f"avg R error: {avg_R_err}\n", file=f)
            
        print("\n\n", file=f)


env.closeEnv()

