import numpy as np
import pybullet as p
import pybullet_data
import time
import keyboard

p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

p.setGravity(0, 0, -10)


plane_id = p.loadURDF("plane.urdf")
armId = p.loadURDF("franka_panda/panda.urdf", [0,0,0], [0,0,0,1], useFixedBase=True)
obj1Id = p.loadURDF("quadruped/quadruped.urdf", [0,0.1,0.3], [0,0,0,1])
numJoints = p.getNumJoints(armId)
end_effector_id = numJoints - 1

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
    img = p.getCameraImage(1000, 1000, view_matrix, projection_matrix)
    return img

# Main loop
while True:
    p.stepSimulation()

    if keyboard.is_pressed('space') or True:
        get_camera_snapshot()
    
    time.sleep(1./240.)