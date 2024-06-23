import pybullet as p
import pybullet_data
import numpy as np
import time
from datetime import datetime
from scipy.spatial.transform import Rotation
from PIL import Image
from math import tan, radians
import pickle
    

class FrankaArmEnvironment:

    def __init__(self, videoLogging=False, out_dir=None):
        p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setRealTimeSimulation(0)
        p.setGravity(0, 0, -10)
        # p.configureDebugVisualizer(p.COV_ENABLE_GUI,0)

        self.videoLogging = videoLogging and (out_dir is not None)
        if self.videoLogging:
            timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            self.loggerId = p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, out_dir + f"\\videolog-{timestamp}.mp4")

        self.startEnv()


    def startEnv(self):
        #set object variables

        self.planeId = p.loadURDF("plane.urdf")
        self.robotId = p.loadURDF("franka_panda/panda.urdf", [0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0], useFixedBase=True)
        self.tableId = p.loadURDF("table/table.urdf", [0.6, 0.0, -0.2], p.getQuaternionFromEuler([0.0, 0.0, np.pi/2]), useFixedBase=True)

        # self.objectId = p.loadURDF("urdf/mug.urdf", [0.5, 0.0, 0.45], p.getQuaternionFromEuler([0.0, 0.0, -np.pi/6]))
        # self.objectId = p.loadURDF("lego/lego.urdf", [0.5, 0.05, 0.45], p.getQuaternionFromEuler([0.0, 0.0, np.pi/3]))
        self.objectId = p.loadURDF("jenga/jenga.urdf", [0.5, 0.0, 0.45], p.getQuaternionFromEuler([0.0, 0.0, np.pi/3]))

        # self.objectId = p.loadURDF("sphere_small.urdf", [0.5, 0.07, 0.45], p.getQuaternionFromEuler([0.0, 0.0, np.pi/3]))
        # p.changeVisualShape(self.objectId, -1, rgbaColor=[0, 1, 1, 1])
        
        # for i in range(10):
        #     p.loadURDF("domino/domino.urdf", [0.55 + i/25, 0.07 - i/25, 0.45], p.getQuaternionFromEuler([0.0, 0.0, -np.pi/4]))


        self.eefDebugLines = [[-1,(1,0,0),[1,0,0]], [-1,(0,1,0),[0,1,0]], [-1,(0,0,1),[0,0,1]]] #list of [id, dir, colour]  pos is fixed to eef position
        self.staticDebugLines = [] #list of [id, pos, dir, colour]
        self.dynamicDebugLines = [] #list of [id, pos, offset, dir, colour]

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
        self.gripperClosed = True

        self.useNullSpace = True

        self.imgSize = 1024
        self.fov = 60
        self.aspect = 1.0
        self.nearplane = 0.01
        self.farplane = 10
        self.projectionMatrix = p.computeProjectionMatrixFOV(self.fov, self.aspect, self.nearplane, self.farplane)
        self.focalLength = self.imgSize / (2 * tan(radians(self.fov) / 2))
        self.principalPoint = (self.imgSize / 2, self.imgSize / 2) #principal point in center of image

        self.setDebugCameraState(1.55, 60.0, -35.0, [0.0, 0.2, 0.0])

        #Let the environment come to rest before starting
        for _ in range(250):
            self.stepEnv()

        self.restPos, self.restOrn = self.robotGetEefPosition()


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
        desiredAngles = desiredAngles[:self.numControlledJoints] 
        #If too many joints given, ignore the extras. We should only set the angles for the 7 joints we control.
        #The fixed joints should be ignored and gripper joints handled differenty
        currAngles = self.robotGetJointAngles()

        for i in range(interpolationSteps):
            alpha = (i+1) / interpolationSteps
            interpolatedPosition = [(1 - alpha) * prev + alpha * desired for prev, desired in zip(currAngles, desiredAngles)]
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
            #step env and interpolation are needed to make the movement smooth and not try to snap
            # to the desired angles as quickly as possible.

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
        currentOrn = self.getMatrixFromQuaternion(orn)
        transformToLocal = np.linalg.inv(currentOrn)

        localRotation = transformToLocal @ rotationMatrix @ currentOrn

        newOrn = currentOrn @ localRotation
        newOrn = self.getQuaternionFromMatrix(newOrn)

        newPos = (np.array(pos) + np.array(translation)).tolist()

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


    def robotGetCameraViewMatrix(self):
        pos, orn = self.robotGetEefPosition()
        rotationMatrix = self.getMatrixFromQuaternion(orn)

        # Initial vectors
        initCameraVector = (0, 0, 1)  # Z axis
        initUpVector = (0, 1, 0)      # Y axis
        # Rotated vectors
        cameraVector = rotationMatrix.dot(initCameraVector)
        upVector = rotationMatrix.dot(initUpVector)
        return p.computeViewMatrix(pos, pos + 0.1 * cameraVector, upVector)


    def robotGetCameraSnapshot(self):
        """
        Get camera snapshot as mounted on end effector
        Returns: (width, height, rgb, depthBuffer, segmentation)
          width, height :: int, of all images
          rgba :: numpy array of rgba values 0-255
          depthBuffer :: numpy array of depth proportions 0-1. 0 corresponds to near plane, 1 corresponds to far plane
          segmentation :: numpy array of segmentation map. Values range over all object ids. (0-num objects in simulation)
        """
        viewMatrix = self.robotGetCameraViewMatrix()

        width, height, rgbPixels, depthPixels, segmentationBuffer = p.getCameraImage(self.imgSize, self.imgSize, viewMatrix, self.projectionMatrix)
        rgba = np.array(rgbPixels).reshape((width, height, 4)).astype(np.uint8)
        depthBuffer = np.array(depthPixels).reshape((width, height))
        segmentation = np.array(segmentationBuffer).reshape((width, height))
        
        return (width, height, rgba, depthBuffer, segmentation, viewMatrix)


    def robotSaveCameraSnapshot(self, filename, path="", rgb=None, depthBuffer=None, vm=None):
        """
        Filename should NOT include a file extension
        Path: Optional relative path to folder
        Can pass rgb AND depth AND vm or automatically call robotGetCameraSnapshot if not provided
          If any are not provided, all will be overwritten with an internal call
        """
        if (rgb is None) or (depthBuffer is None):
            _, _, rgb, depthBuffer, _, vm = self.robotGetCameraSnapshot()

        rgbImg = Image.fromarray(rgb)
        rgbImg = rgbImg.convert("RGB") #RGB, no alpha channel
        rgbImg.save(f"{path}\\{filename}-rgb.jpg")

        with open(f"{path}\\{filename}-depth.pkl", 'wb') as f:
            pickle.dump(depthBuffer, f)

        with open(f"{path}\\{filename}-vm.pkl", 'wb') as f:
            pickle.dump(vm, f)
    

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
        _, _, _, _, _, _, _, _, cameraYaw, cameraPitch, cameraDist, cameraTarget = p.getDebugVisualizerCamera()
        #For some reason pybullet getDebugCamera returns values in a differnt order to what setDebugCamera takes in.
        # The env wrapper fixes this, but its not a bug, the order of values is correct even though it looks weird
        return (cameraDist, cameraYaw, cameraPitch, cameraTarget)

    def setDebugCameraState(self, cameraDist, cameraYaw, cameraPitch, cameraTarget=[0,0,0]):
        p.resetDebugVisualizerCamera(cameraDist, cameraYaw, cameraPitch, cameraTarget)


    def addDebugLine(self, pos, direction, colour, static):
        """
        Add a debug line to be drawn each frame
        if static is True:
          line is drawn fixed at world coords pos, with direction and colour as specified
        else:
          line is drawn at pos offset from eef position, with direction and colour as specified
        colour is a list [R,G,B] with values 0..1
        """
        if static:
            self.staticDebugLines.append([-1, pos, direction, colour])
        else:
            offset = np.zeros(3)
            self.dynamicDebugLines.append([-1, pos, offset, direction, colour])


    def removeAllDebugLines(self):
        """
        Removes all drawn static and dynamic debug lines.
        DOES NOT remove eefDebugLines, since these should always be drawn
        """
        for line in self.staticDebugLines + self.dynamicDebugLines:
            p.removeUserDebugItem(line[0])
        self.staticDebugLines = []
        self.dynamicDebugLines = []


    def drawDebugLines(self):
        """
        Draws the lines specified in self.eefDebugLines, self.staticDebugLines and self.dynamicDebugLines
        """
        pos, orn = self.robotGetEefPosition()
        rotationMatrix = self.getMatrixFromQuaternion(orn)

        for line in self.eefDebugLines:
            lineId, direction, colour = line
            lineVector = rotationMatrix.dot(direction)
            stop = pos + 0.2 * lineVector
            line[0] = p.addUserDebugLine(pos, stop, colour, replaceItemUniqueId=lineId)

        for line in self.staticDebugLines:
            lineId, start, direction, colour = line
            end = start + 0.05 * np.array(direction)
            line[0] = p.addUserDebugLine(start, end, colour, replaceItemUniqueId=lineId)

        for line in self.dynamicDebugLines:
            lineId, start, offset, direction, colour = line

            offset = (np.array(pos) - np.array(self.restPos))

            end = start + offset + 0.05 * np.array(direction)
            line[0] = p.addUserDebugLine(start + offset, end, colour, replaceItemUniqueId=lineId)
            line[2] = offset


    def disableHUD(self):
        p.configureDebugVisualizer(p.COV_ENABLE_GUI,0)


    def enableHud(self):
        p.configureDebugVisualizer(p.COV_ENABLE_GUI,1)




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
