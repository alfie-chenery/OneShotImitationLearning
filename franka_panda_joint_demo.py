import pybullet as p
import time
import pybullet_data
import math
import random

p.connect(p.GUI)#or p.DIRECT for non-graphical version
p.resetSimulation()
p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
p.setGravity(0,0,-10)

planeId = p.loadURDF("plane.urdf")
startOrientation = p.getQuaternionFromEuler([0,0,0])
armId = p.loadURDF("franka_panda/panda.urdf", [0,0,0], startOrientation, useFixedBase=True)
numJoints = p.getNumJoints(armId)
end_effector_id = numJoints - 1

rest = [0] * numJoints
steps = 500
for i in range(numJoints):
    jointInfo = p.getJointInfo(armId, i)
    rest[i] = jointInfo[8] + (jointInfo[9] - jointInfo[8]) / 2 #center of lower and upper
    print(f"joint {i}: {jointInfo}\n")

for i in range (numJoints):
    print(f"Rotating Joint {i}")
    for t in range(500):
        jointPoses = rest.copy()
        jointInfo = p.getJointInfo(armId, i)
        lower = jointInfo[8]
        upper = jointInfo[9]
        jointPoses[i] = lower + t * (upper/500)
        forces=[500.0]*len(jointPoses)

        p.setJointMotorControlArray(armId, 
                                    range(len(jointPoses)),
                                    p.POSITION_CONTROL,
                                    targetPositions=jointPoses,
                                    forces=forces)




        p.stepSimulation()
        time.sleep(1./240.)


p.disconnect()
