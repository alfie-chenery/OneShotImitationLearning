#Install this DINO repo to extract correspondences: https://github.com/ShirAmir/dino-vit-features

import numpy as np
import torch
from dinofeatures.correspondences import find_correspondences

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



#Hyperparameters for DINO correspondences extraction
num_pairs = 8
load_size = 224
layer = 9
facet = 'key'
bin=True
thresh=0.05
model_type='dino_vitb8'
stride=4

if __name__ == "__main__":
    
 #Get rgbd from wrist camera.
    rgb_bn, depth_bn = camera.get_rgbd()
    Error = 100000
    ERR_THRESHOLD = 50 #A generic error between the two sets of points
    while error > ERR_THRESHOLD:
        rgb_live, depth_live = camera.get_rgbd()
        with torch.no_grad():
            points1, points2, image1_pil, image2_pil = find_correspondences(rgb_live, rgb_bn, num_pairs, load_size, layer,
                                                                               facet, bin, thresh, model_type, stride)
            #Given the pixel coordinates of the correspondences, add the depth channel
            points1 = add_depth(points1, depth_bn)
            points2 = add_depth(points2, depth_live)
            R, t = find_transformation(points1, points2)

            #A function to convert pixel distance into meters based on calibration of camera.
            t_meters = convert_pixels_to_meters(t)

            #Move robot
            robot.move(t_meters,R)
            error = compute_error(points1, points2)