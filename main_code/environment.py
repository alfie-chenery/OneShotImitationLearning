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
        self.planeId = p.loadURDF("plane.urdf")
        self.robotId = p.loadURDF("franka_panda/panda.urdf", [0,0,0], [0,0,0,1], useFixedBase=True)
        self.obj1Id = p.loadURDF("quadruped/quadruped.urdf", [0,0.1,0.3], [0,0,0,1])

        self.numJoints = p.getNumJoints(self.robotId)
        self.fixed_joints = [7,8,11] #These joints are fixed and cannot move
        self.eefId = self.numJoints - 1
        self.dof = self.numJoints - len(self.fixed_joints)

        self.imgSize = 1000
        self.fov = 60 #degrees?
        self.aspect = 1.0     # leave as 1 for square image
        self.nearplane = 0.01 # wtf are these units? gotta figure that out
        self.farplane = 100
        self.projectionMatrix = p.computeProjectionMatrixFOV(self.fov, self.aspect, self.nearplane, self.farplane)
        print("Projection matrix")
        print(np.array(self.projectionMatrix).reshape((4,4)))
        print("")


    def stepEnv(self):
        p.stepSimulation()
        time.sleep(1./240.)


    def closeEnv(self):
        #save relevant stuff
        p.disconnect()


    def robotGetJointAngles(self):
        return [p.getJointState(self.robotId, i)[0] for i in range(self.numJoints)]
    

    def robotSetJointAngles(self, desiredAngles, interpolationSteps=100):
        #Desired angles should be a list of size 12. Even though joints 7,8,11 are fixed and the values
        # will be ignored, the list needs to be size 12
        for i in range(interpolationSteps):
            alpha = (i+1) / interpolationSteps
            interpolatedPosition = [(1 - alpha) * prev + alpha * desired for prev, desired in zip(self.robotGetJointAngles(), desiredAngles)]
            forces = [500.0] * len(interpolatedPosition)

            p.setJointMotorControlArray(self.robotId, 
                                        range(len(interpolatedPosition)),
                                        p.POSITION_CONTROL,
                                        targetPositions=interpolatedPosition,
                                        forces=forces)
            
            #self.stepEnv()
            #step env is probably needed. Its definatley needed to see the movement, but i wonder if
            # setting to the goal state is actually fine and it moves smoothly or if the robot will just
            # snap there. If we remove this and let the environment step only in the main loop,
            # then interpolation steps arent needed

            #could also make the environment store a buffer of desired positions and step through them,
            # this would let us fill it with the interpolated steps, and the movements happen only
            # when env.step happens in the main loop. It feels kind of desireable to have env.step only
            # in the main loop, since otherwise these movement functions are blocking until its done moving


    def robotGetEefPosition(self):
        pos, orn, _, _, _, _ = p.getLinkState(self.robotId, self.eefId, computeForwardKinematics=True)
        return pos, orn


    def robotSetEefPosition(self, pos, orn):
        jointAngles = p.calculateInverseKinematics(self.robotId, self.eefId, pos, orn)

        for i in self.fixed_joints:
            jointAngles.insert(i, 0)
        #inverse kinematics returns a list of size dof (for this robot 9)
        # but when setting joint angles we expect to set every joint (all 12, even though the fixed ones will be ignored)
        # easiest to add dummy values for these joints before setting angles, they will be ignored anyway

        self.robotSetJointAngles(jointAngles)


    def robotMoveEefPosition(self, translation, rotationMatrix):
        pos, orn = self.robotGetEefPosition()
        ornMatrix = np.array(p.getMatrixFromQuaternion(orn)).reshape((3,3))
        self.robotSetEefPosition(pos + translation, np.dot(ornMatrix, rotationMatrix))
        
        #might be a problem. orn is now a 3x3 rotation matrix but I think the later code is expecting a quaternion


    def robotGetCameraSnapshot(self):
        pos, orn = self.robotGetEefPosition()
        rotationMatrix = np.array(p.getMatrixFromQuaternion(orn)).reshape((3, 3))

        print("Rotation matrix")
        print(rotationMatrix)
        print("")

        # Initial vectors
        initCameraVector = (0, 0, 1) # z-axis
        initUpVector = (0, 1, 0) # y-axis
        # Rotated vectors
        cameraVector = rotationMatrix.dot(initCameraVector)
        upVector = rotationMatrix.dot(initUpVector)
        viewMatrix = p.computeViewMatrix(pos, pos + 0.1 * cameraVector, upVector)

        print("View matrix")
        print(np.array(viewMatrix).reshape((4,4)))
        print("")

        width, height, rgbPixels, depthPixels, segmentationBuffer = p.getCameraImage(self.imgSize, self.imgSize, viewMatrix, self.projectionMatrix)
        rgb = np.array(rgbPixels).reshape((width, height, 4))
        depth = np.array(depthPixels).reshape((width, height))
        depth = self.farplane * self.nearplane / (self.farplane - (self.farplane - self.nearplane) * depth)
        segmentation = np.array(segmentationBuffer).reshape((width, height))
        segmentation = segmentation * 1.0 / 255.0
        
        return (width, height, rgb, depth, segmentation)


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
