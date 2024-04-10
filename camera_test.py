import numpy as np
import pybullet as p
import pybullet_data
import time
import keyboard
from matplotlib import pyplot as plt
from PIL import Image

p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -10)

plane_id = p.loadURDF("plane.urdf")
armId = p.loadURDF("franka_panda/panda.urdf", [0,0,0], [0,0,0,1], useFixedBase=True)
obj1Id = p.loadURDF("quadruped/quadruped.urdf", [0,0.1,0.3], [0,0,0,1])
numJoints = p.getNumJoints(armId)
end_effector_id = numJoints - 1

img_size = 1000
fov, aspect, nearplane, farplane = 60, 1.0, 0.01, 100
projection_matrix = p.computeProjectionMatrixFOV(fov, aspect, nearplane, farplane)

def get_camera_snapshot():
    pos, orn, _, _, _, _ = p.getLinkState(armId, end_effector_id, computeForwardKinematics=True)
    rotation_matrix = p.getMatrixFromQuaternion(orn)
    rotation_matrix = np.array(rotation_matrix).reshape(3, 3)
    # Initial vectors
    init_camera_vector = (0, 0, 1) # z-axis
    init_up_vector = (0, 1, 0) # y-axis
    # Rotated vectors
    camera_vector = rotation_matrix.dot(init_camera_vector)
    up_vector = rotation_matrix.dot(init_up_vector)
    view_matrix = p.computeViewMatrix(pos, pos + 0.1 * camera_vector, up_vector)
    img = p.getCameraImage(img_size, img_size, view_matrix, projection_matrix)
    return img

def save_snapshot(img):
    #img = p.getCameraImage(224, 224, shadow = False, renderer=p.ER_BULLET_HARDWARE_OPENGL)
    rgb_opengl = (np.reshape(img[2], (img_size, img_size, 4)))
    depth_buffer_opengl = np.reshape(img[3], [img_size, img_size])
    depth_opengl = farplane * nearplane / (farplane - (farplane - nearplane) * depth_buffer_opengl)
    seg_opengl = np.reshape(img[4], [img_size, img_size]) * 1. / 255.

    rgbim = Image.fromarray(rgb_opengl)
    rgbim_no_alpha = rgbim.convert('RGB')

    rgbim_no_alpha.save('rgb.jpg')
    plt.imsave('depth.jpg', depth_buffer_opengl)

# Main loop
while True:
    p.stepSimulation()

    if keyboard.is_pressed('space'):
        img = get_camera_snapshot()
        save_snapshot(img)
    
    time.sleep(1./240.)