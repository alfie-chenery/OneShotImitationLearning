import pybullet as p
import time
import pybullet_data
import math
import random
import keyboard

p.connect(p.GUI)#or p.DIRECT for non-graphical version
p.resetSimulation()
p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
p.setGravity(0,0,-10)

planeId = p.loadURDF("plane.urdf")
blockId = p.loadURDF("lego/lego.urdf",[random.uniform(-2.0,2.0), random.uniform(-2.0,2.0), 0.5])
startOrientation = p.getQuaternionFromEuler([0,0,0])
armId = p.loadURDF("franka_panda/panda.urdf", [0,0,0], startOrientation, useFixedBase=True)
numJoints = p.getNumJoints(armId)
end_effector_id = numJoints - 1

ignore_joints = [7,8,11] #these joints are fixed and so nverse kinematics ignores them
# in fact it returns a list of 9 elements because we only have 9 degrees of freedom not 12
links = [0,1,2,3,4,5,6,9,10]

for i in range(numJoints):
    print(f"{i}: {p.getJointInfo(armId, i)[8]}, {p.getJointInfo(armId, i)[9]}")


for i in range (10000):

    pos, _ = p.getBasePositionAndOrientation(blockId)
    orn = p.getQuaternionFromEuler([1,0,0])
    jointPoses = list(p.calculateInverseKinematics(armId, end_effector_id, pos, orn))
    
    for i in ignore_joints:
        jointPoses.insert(i, 0)

    forces=[500.0]*len(jointPoses)

    p.setJointMotorControlArray(armId, 
                                range(len(jointPoses)),
                                p.POSITION_CONTROL,
                                targetPositions=jointPoses,
                                forces=forces)




    p.stepSimulation()
    time.sleep(1./240.)


p.disconnect()
