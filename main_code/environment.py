import pybullet as p
import pybullet_data
import numpy as np
import time
    

class FrankaArmEnvironment:

    def __init__(self):
        p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -10)
        self.startEnv()


    def startEnv(self):
        #set object variables
        self.robotId = p.loadURDF("franka_panda/panda.urdf", [0,0,0], [0,0,0,1], useFixedBase=True)
        self.planeId = p.loadURDF("plane.urdf")
        self.eefId = p.getNumJoints(self.robotId) - 1
        self.obj1Id = p.loadURDF("quadruped/quadruped.urdf", [0,0.1,0.3], [0,0,0,1])

        self.img_size = 1000
        self.fov = 60
        self.aspect = 1.0
        self.nearplane = 0.01
        self.farplane = 100
        self.projection_matrix = p.computeProjectionMatrixFOV(self.fov, self.aspect, self.nearplane, self.farplane)


    def stepEnv(self):
        p.stepSimulation()
        time.sleep(1./240.)


    def closeEnv(self):
        #save relevant stuff
        #close simulation
        pass


    def robotGetCameraSnapshot(self):
        pos, orn, _, _, _, _ = p.getLinkState(self.robotId, self.eefId, computeForwardKinematics=True)
        rotation_matrix = p.getMatrixFromQuaternion(orn)
        rotation_matrix = np.array(rotation_matrix).reshape(3, 3)
        # Initial vectors
        init_camera_vector = (0, 0, 1) # z-axis
        init_up_vector = (0, 1, 0) # y-axis
        # Rotated vectors
        camera_vector = rotation_matrix.dot(init_camera_vector)
        up_vector = rotation_matrix.dot(init_up_vector)
        view_matrix = p.computeViewMatrix(pos, pos + 0.1 * camera_vector, up_vector)
        img = p.getCameraImage(self.img_size, self.img_size, view_matrix, self.projection_matrix)
        return img


    # def save_snapshot(img):
    #     #img = p.getCameraImage(224, 224, shadow = False, renderer=p.ER_BULLET_HARDWARE_OPENGL)
    #     rgb_opengl = (np.reshape(img[2], (img_size, img_size, 4)))
    #     depth_buffer_opengl = np.reshape(img[3], [img_size, img_size])
    #     depth_opengl = farplane * nearplane / (farplane - (farplane - nearplane) * depth_buffer_opengl)
    #     seg_opengl = np.reshape(img[4], [img_size, img_size]) * 1. / 255.

    #     rgbim = Image.fromarray(rgb_opengl)
    #     rgbim_no_alpha = rgbim.convert('RGB')

    #     rgbim_no_alpha.save('rgb.jpg')
    #     plt.imsave('depth.jpg', depth_buffer_opengl)
