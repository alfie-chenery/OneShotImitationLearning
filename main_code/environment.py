import pybullet as p
import pybullet_data
import numpy as np
import time
from quat_utils import quaternion_from_matrix
from PIL import Image

    

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
        self.tableId = p.loadURDF("table/table.urdf", [0.6,0,-0.2], p.getQuaternionFromEuler([0,0,np.pi/2]))
        self.objectId = p.loadURDF("urdf/mug.urdf", [0.8, 0, 0.5], [0,0,0,1])

        self.numJoints = p.getNumJoints(self.robotId)
        self.fixed_joints = [7,8,11] #These joints are fixed and cannot move
        self.eefId = self.numJoints - 1
        self.dof = self.numJoints - len(self.fixed_joints)

        self.lowerLimits = []
        self.upperLimits = []
        self.jointRanges = []
        self.restPoses = [-0.0016406124581102627, 0.026211667016220668, 0.002989846306765971, -0.9495545304254569, -5.895048550690606e-05, 1.2393585715718878, -1.7747677120392198e-05, 0.0, 0.0, -9.177073744494914e-20, 0.00065723506632391, 0.0]
        for i in range(self.numJoints):
            jointInfo = p.getJointInfo(self.robotId, i)
            self.lowerLimits.append(jointInfo[8])
            self.upperLimits.append(jointInfo[9])
            self.jointRanges.append(jointInfo[9] - jointInfo[8])
            
            p.resetJointState(self.robotId, i, self.restPoses[i])
        self.useNullSpace = False

        self.imgSize = 224
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
            
            self.stepEnv()
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


    def robotSetEefPosition(self, pos, orn=None):
        if self.useNullSpace:
            jointAngles = list(p.calculateInverseKinematics(self.robotId, self.eefId, pos, orn, self.lowerLimits, self.upperLimits, self.jointRanges, self.restPoses))
        else:
            jointAngles = list(p.calculateInverseKinematics(self.robotId, self.eefId, pos, orn))

        for i in self.fixed_joints:
            jointAngles.insert(i, 0)
        #inverse kinematics returns a list of size dof (for this robot 9)
        # but when setting joint angles we expect to set every joint (all 12, even though the fixed ones will be ignored)
        # easiest to add dummy values for these joints before setting angles, they will be ignored anyway

        self.robotSetJointAngles(jointAngles)


    def robotMoveEefPosition(self, translation, rotationMatrix):
        pos, orn = self.robotGetEefPosition()
        posVec = np.array(pos)
        ornMatrix = np.array(p.getMatrixFromQuaternion(orn)).reshape((3,3))

        print(f"Rotation: {orn}, {ornMatrix}")

        self.robotSetEefPosition(posVec + translation, quaternion_from_matrix(np.dot(ornMatrix, rotationMatrix)))
        
        #find a module to do the matrix to quat for me. ChatGPT solution seems dodgy
        # removing the need for quat_utils file


    def robotGetCameraSnapshot(self):
        """
        Get camera snapshot as mounted on end effector
        Returns: (width, height, rgb, depthBuffer, segmentation)
          width, height :: int, of all images
          rgb :: numpy array of rgb values 0-255
          depthBuffer :: numpy array of depth proportions 0-255. To calculate actual depth pass to env.calculateDepthFromBuffer()
          segmentation :: numpy array of segmentation map   TODO: work out the range of values, i dont actually use this anywhere yet
        """
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
        rgb = np.array(rgbPixels).reshape((width, height, 4)).astype(np.uint8)
        depthBuffer = np.array(depthPixels).reshape((width, height))
        depthBuffer = (depthBuffer * 255).astype(np.uint8)
        segmentation = np.array(segmentationBuffer).reshape((width, height))
        segmentation = segmentation * 1.0 / 255.0
        
        return (width, height, rgb, depthBuffer, segmentation)


    def robotSaveCameraSnapshot(self, filename, path="", rgb=None, depthBuffer=None):
        """
        Filename should NOT include a file extension
        Path: Optional relative path to folder
        Can pass rgb AND depth buffer or automatically call robotGetCameraSnapshot if not provided
          If either is not provided, both will be overwritten with an internal call
        """
        if (rgb is None) or (depthBuffer is None):
            _, _, rgb, depthBuffer, _ = self.robotGetCameraSnapshot()
        
        #timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

        rgbImg = Image.fromarray(rgb)
        rgbImg = rgbImg.convert("RGB") #RGB, no alpha channel

        depthImg = Image.fromarray(depthBuffer)
        depthImg = depthImg.convert("L") #Luminosity (single greyscale channel) no alpha

        rgbImg.save(f"{path}\\{filename}-rgb.jpg")
        depthImg.save(f"{path}\\{filename}-depth.jpg")

    
    def calculateDepthFromBuffer(self, depthBuffer):
        """
        Convert depth buffer (a 2d numpy array of image values 0-255) to an
        array of the same size, with values which store the actual depth values
        """
        depthBuffer = depthBuffer.astype(np.float32) * 1.0 / 255.0
        depth = self.farplane * self.nearplane / (self.farplane - (self.farplane - self.nearplane) * depthBuffer)
        return depth


    def setDebugCameraPos(self, cameraDist, cameraYaw, cameraPitch):
        p.resetDebugVisualizerCamera(cameraDist, cameraYaw, cameraPitch, [0,0,0])


    def enableWireframe(self):
        p.configureDebugVisualizer(p.COV_ENABLE_WIREFRAME, 1)


    def disableWireframe(self):
        p.configureDebugVisualizer(p.COV_ENABLE_WIREFRAME, 0)



#Testing
if __name__ == "__main__":
    env = FrankaArmEnvironment()
    targetPosXId = p.addUserDebugParameter("targetPosX",-1,1,0.5)
    targetPosYId = p.addUserDebugParameter("targetPosY",-1,1,0)
    targetPosZId = p.addUserDebugParameter("targetPosZ",-1,1,1)
    nullSpaceId = p.addUserDebugParameter("nullSpace",0,1,1)
    targetOrnRollId = p.addUserDebugParameter("targetOrnRoll",-np.pi,np.pi,np.pi)
    targetOrnPitchId = p.addUserDebugParameter("targetOrnPitch",-np.pi,np.pi,0)
    targetOrnYawId = p.addUserDebugParameter("targetOrnYaw",-np.pi,np.pi,0)

    while True:
        targetPosX = p.readUserDebugParameter(targetPosXId)
        targetPosY = p.readUserDebugParameter(targetPosYId)
        targetPosZ = p.readUserDebugParameter(targetPosZId)
        nullSpace = p.readUserDebugParameter(nullSpaceId)
        targetOrnRoll = p.readUserDebugParameter(targetOrnRollId)
        targetOrnPitch = p.readUserDebugParameter(targetOrnPitchId)
        targetOrnYaw = p.readUserDebugParameter(targetOrnYawId)
    
        targetPosition = [targetPosX,targetPosY,targetPosZ]
        targetOrn = [targetOrnRoll,targetOrnPitch,targetOrnYaw]
        env.useNullSpace = nullSpace > 0.5
        env.robotSetEefPosition(targetPosition, p.getQuaternionFromEuler(targetOrn))
        env.stepEnv()
