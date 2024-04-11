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
        self.numJoints = p.getNumJoints(self.robotId)
        self.eefId = self.numJoints - 1
        self.obj1Id = p.loadURDF("quadruped/quadruped.urdf", [0,0.1,0.3], [0,0,0,1])

        self.imgSize = 1000
        self.fov = 60
        self.aspect = 1.0
        self.nearplane = 0.01
        self.farplane = 100
        self.projectionMatrix = p.computeProjectionMatrixFOV(self.fov, self.aspect, self.nearplane, self.farplane)


    def stepEnv(self):
        p.stepSimulation()
        time.sleep(1./240.)


    def closeEnv(self):
        #save relevant stuff
        p.disconnect()


    def getJointAngles(self):
        return [p.getJointState(self.robotId, i)[0] for i in range(self.numJoints)]
    

    def setJointAngles(self, desiredAngles, interpolationSteps=100):
        for i in range(interpolationSteps):
            alpha = (i+1) / interpolationSteps
            interpolatedPosition = [(1 - alpha) * prev + alpha * desired for prev, desired in zip(self.getJointAngles(), desiredAngles)]
            forces = [500.0] * len(interpolatedPosition)

            p.setJointMotorControlArray(self.robotId, 
                                        range(len(interpolatedPosition)),
                                        p.POSITION_CONTROL,
                                        targetPositions=interpolatedPosition,
                                        forces=forces)
            
            #self.stepEnv()


    def robotGetCameraSnapshot(self):
        pos, orn, _, _, _, _ = p.getLinkState(self.robotId, self.eefId, computeForwardKinematics=True)
        rotationMatrix = p.getMatrixFromQuaternion(orn)
        rotationMatrix = np.array(rotationMatrix).reshape(3, 3)
        # Initial vectors
        initCameraVector = (0, 0, 1) # z-axis
        initUpVector = (0, 1, 0) # y-axis
        # Rotated vectors
        cameraVector = rotationMatrix.dot(initCameraVector)
        upVector = rotationMatrix.dot(initUpVector)
        viewMatrix = p.computeViewMatrix(pos, pos + 0.1 * cameraVector, upVector)

        img = p.getCameraImage(self.imgSize, self.imgSize, viewMatrix, self.projectionMatrix)
        # img :: ( width::int,
        #          height::int,
        #          rgbPixels::list of [R,G,B,A] [0..width*height],
        #          depthPixels::list of float [0..width*height],
        #          segmentationMaskBuffer::list of int [0..width*height] )
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


    def setDebugCameraPos(self, cameraDist, cameraYaw, cameraPitch):
        p.resetDebugVisualizerCamera(cameraDist, cameraYaw, cameraPitch, [0,0,0])


    def enableWireframe(self):
        p.configureDebugVisualizer(p.COV_ENABLE_WIREFRAME, 1)


    def disableWireframe(self):
        p.configureDebugVisualizer(p.COV_ENABLE_WIREFRAME, 0)
