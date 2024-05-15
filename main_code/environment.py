import pybullet as p
import pybullet_data
import numpy as np
import time
from datetime import datetime
from scipy.spatial.transform import Rotation
from PIL import Image
    

class FrankaArmEnvironment:

    def __init__(self, videoLogging=False, out_dir=None):
        p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setRealTimeSimulation(0)
        p.setGravity(0, 0, -10)

        self.videoLogging = videoLogging and (out_dir is not None)
        if self.videoLogging:
            timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            self.loggerId = p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, out_dir + f"\\videolog-{timestamp}.mp4")

        self.startEnv()


    def startEnv(self):
        #set object variables

        self.planeId = p.loadURDF("plane.urdf")
        self.robotId = p.loadURDF("franka_panda/panda.urdf", [0, 0, 0], [0, 0, 0, 1], useFixedBase=True)
        self.tableId = p.loadURDF("table/table.urdf", [0.6, 0, -0.2], p.getQuaternionFromEuler([0,0,np.pi/2]), useFixedBase=True)
        #self.objectId = p.loadURDF("urdf/mug.urdf", [0.63, 0.05, 0.45], [0, 0, 0, 1])
        self.objectId = p.loadURDF("lego/lego.urdf", [0.6, 0.05, 0.45], p.getQuaternionFromEuler([0,0,np.pi/3]))
        self.debugLines = [[-1,(1,0,0),[1,0,0]], [-1,(0,1,0),[0,1,0]], [-1,(0,0,1),[0,0,1]]]  #list of [id, vector, colour]

        self.numAllJoints = p.getNumJoints(self.robotId)
        self.eefId = self.numAllJoints - 1
        self.dof = self.numAllJoints - 3        #3 of the joints are not actuated and cannot be controlled
        self.numControlledJoints = self.dof - 2 #Gripper is controlled specially, not by setting its position, so we only expect
                                                # numControlledJoints (7) angles to be passed to the relevant functions

        self.lowerLimits = []
        self.upperLimits = []
        self.jointRanges = []
        self.restPoses = [0.0, 0.0, 0.0, -0.375 * np.pi, 0.0, 0.375 * np.pi, 0.25 * np.pi, 0.0, 0.0, 0.0, 0.0, 0.0]
        for i in range(self.numControlledJoints):
            jointInfo = p.getJointInfo(self.robotId, i)
            self.lowerLimits.append(jointInfo[8])
            self.upperLimits.append(jointInfo[9])
            self.jointRanges.append(jointInfo[9] - jointInfo[8])
            p.resetJointState(self.robotId, i, self.restPoses[i])
        self.restPos, self.restOrn = self.robotGetEefPosition()
        self.gripperClosed = True

        self.useNullSpace = True

        self.imgSize = 224
        self.fov = 60 #degrees?
        self.aspect = 1.0     # leave as 1 for square image
        self.nearplane = 0.01 # wtf are these units? gotta figure that out
        self.farplane = 100
        self.projectionMatrix = p.computeProjectionMatrixFOV(self.fov, self.aspect, self.nearplane, self.farplane)


    def stepEnv(self):
        p.stepSimulation()
        self.drawDebugLines()
        time.sleep(1./240.)


    def resetEnv(self):
        p.resetSimulation()
        self.startEnv()


    def closeEnv(self):
        if self.videoLogging:
            p.stopStateLogging(self.loggerId)

        p.disconnect()


    def robotGetJointAngles(self):
        return [p.getJointState(self.robotId, i)[0] for i in range(self.numControlledJoints)]
    

    def robotSetJointAngles(self, desiredAngles, interpolationSteps=100):
        """
        Desired angles should be a list of size numControlledJoints (7).
        Joints 0-6 are controlled by desiredAngles
        Joints 7,8,11 are fixed and not actuated
        Joints 9,10 are the gripper fingers and are controlled seperately by robotCloseGripper() and robotOpenGripper()
        """
        for i in range(interpolationSteps):
            alpha = (i+1) / interpolationSteps
            interpolatedPosition = [(1 - alpha) * prev + alpha * desired for prev, desired in zip(self.robotGetJointAngles(), desiredAngles)]
            forces = [500.0] * len(interpolatedPosition)

            p.setJointMotorControlArray(self.robotId, 
                                        range(len(interpolatedPosition)),
                                        p.POSITION_CONTROL,
                                        targetPositions=interpolatedPosition,
                                        forces=forces)
            
            velocities = [-1,-1] if self.gripperClosed else [1,1]
            p.setJointMotorControlArray(self.robotId, 
                                        [9,10],
                                        p.VELOCITY_CONTROL,
                                        targetVelocities=velocities,
                                        forces=[10,10])
            
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


    def robotSetEefPosition(self, pos, orn=None, interpolationSteps=100):
        if self.useNullSpace:
            jointAngles = list(p.calculateInverseKinematics(self.robotId, self.eefId, pos, orn, self.lowerLimits, self.upperLimits, self.jointRanges, self.restPoses))
        else:
            jointAngles = list(p.calculateInverseKinematics(self.robotId, self.eefId, pos, orn))

        jointAngles = jointAngles[:self.numControlledJoints]
        #inverse kinematics returns a list of size dof (for this robot 9)
        # but when setting joint angles we only set the numControlledJoints (7) joints of the arm
        # the two joints of the gripper are handled differently

        self.robotSetJointAngles(jointAngles, interpolationSteps=interpolationSteps)


    def robotMoveEefPosition(self, translation, rotationMatrix, interpolationSteps=100):
        pos, orn = self.robotGetEefPosition()
        
        # quat = self.getQuaternionFromMatrix(rotationMatrix)
        # newPos, newOrn = p.multiplyTransforms(pos, orn, translation, quat)

        newPos, newOrn = self.offsetMovementLocal(pos, orn, translation, rotationMatrix)

        self.robotSetEefPosition(newPos, newOrn, interpolationSteps=interpolationSteps)


    def offsetMovementLocal(self, pos, orn, translation, rotationMatrix):
        """
        Take pos and orn and add translation and rotate by rotationMatrix in local space, about the current position (not about the world origin)
        """
        newPos = [p + t for (p,t) in zip(pos, translation)]
        ornMat = np.array(p.getMatrixFromQuaternion(orn)).reshape((3,3))
        newOrnMat = np.dot(ornMat, rotationMatrix)
        newOrn = self.getQuaternionFromMatrix(newOrnMat)

        return (newPos, newOrn)
    

    def calculateOffset(self, posA, ornA, posB, ornB):
        """
        Calculate offset from posA, ornA to posB, ornB
        """
        # dPos, dOrn = p.invertTransform(posB, ornB)
        # return p.multiplyTransforms(posA, ornA, dPos, dOrn)
        
        # dPos, dOrn = p.invertTransform(posA, ornA)
        # return p.multiplyTransforms(posB, ornB, dPos, dOrn)
    
        dPos = [b-a for (a,b) in zip(posA, posB)]
        dOrn = p.getDifferenceQuaternion(ornB, ornA)
        return (dPos, dOrn)


    def robotCloseGripper(self):
        self.gripperClosed = True


    def robotOpenGripper(self):
        self.gripperClosed = False


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

        # Initial vectors
        initCameraVector = (0, 0, 1) # z-axis
        initUpVector = (0, 1, 0) # y-axis
        # Rotated vectors
        cameraVector = rotationMatrix.dot(initCameraVector)
        upVector = rotationMatrix.dot(initUpVector)
        viewMatrix = p.computeViewMatrix(pos, pos + 0.1 * cameraVector, upVector)

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
        calculated from camera calibration
        """
        depthBuffer = depthBuffer.astype(np.float32) * 1.0 / 255.0
        depth = self.farplane * self.nearplane / (self.farplane - (self.farplane - self.nearplane) * depthBuffer)
        return depth
    

    def pixelsToMetres(self, pixels):
        """
        Convert distance measured in pixels of images to distance in metres
        based on the calibration of the camera pixel density
        """
        #Some sources say 1 pixel is 1mm. Seems correct
        return pixels * 0.001
    

    def getQuaternionFromEuler(self, e):
        #Expose pybullet conversion functions so main code doesnt need to import pybullet itself
        return p.getQuaternionFromEuler(e)
    
    def getEulerFromQuaternion(self, q):
        #Expose pybullet conversion functions so main code doesnt need to import pybullet itself
        return p.getEulerFromQuaternion(q)
    
    def getMatrixFromQuaternion(self, q):
        #Expose pybullet conversion functions so main code doesnt need to import pybullet itself
        return np.array(p.getMatrixFromQuaternion(q)).reshape((3,3))
    
    def getQuaternionFromMatrix(self, m):
        #Pybullet doesnt have all the necessary conversion functions. Because it couldnt be easy could it
        rotation = Rotation.from_matrix(m)
        return rotation.as_quat().tolist()
    
    def getMatrixFromEuler(self, e):
        return self.getMatrixFromQuaternion(self.getQuaternionFromEuler(e))
    
    def getEulerFromMatrix(self, m):
        return self.getEulerFromQuaternion(self.getQuaternionFromMatrix(m))


    def getDebugCameraState(self):
        _, _, _, _, _, _, _, _, cameraDist, cameraYaw, cameraPitch, cameraTarget = p.getDebugVisualizerCamera()
        return (cameraDist, cameraYaw, cameraPitch, cameraTarget)

    def setDebugCameraState(self, cameraDist, cameraYaw, cameraPitch, cameraTarget=[0,0,0]):
        p.resetDebugVisualizerCamera(cameraDist, cameraYaw, cameraPitch, cameraTarget)

    
    def drawDebugLines(self):
        """
        Draws the lines specified in self.debugLines
        Each line is of the form [id, vector, colour]
        where vector is relative to the robots eef and colour is a list [R,G,B] with values 0..1
        """
        start, orn = self.robotGetEefPosition()
        rotationMatrix = np.array(p.getMatrixFromQuaternion(orn)).reshape((3, 3))

        for line in self.debugLines:
            lineVector = rotationMatrix.dot(line[1])
            stop = start + 0.2 * lineVector

            line[0] = p.addUserDebugLine(start, stop, line[2], replaceItemUniqueId=line[0])




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
