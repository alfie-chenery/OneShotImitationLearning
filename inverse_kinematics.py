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
blockId = p.loadURDF("lego/lego.urdf",[random.uniform(-2.0,2.0), random.uniform(-2.0,2.0), 0.5])
startOrientation = p.getQuaternionFromEuler([0,0,0])
armId = p.loadURDF("franka_panda/panda.urdf", [0,0,0], startOrientation, useFixedBase=True)
numJoints = p.getNumJoints(armId)
end_effector_id = numJoints - 1

for i in range (10000):
    pos, _ = p.getBasePositionAndOrientation(blockId)
    orn = [0,0,0,1]
    jointPoses = p.calculateInverseKinematics(armId, end_effector_id, pos, orn)
    forces=[500.0]*len(jointPoses)

    p.setJointMotorControlArray(armId, 
                                range(len(jointPoses)),
                                p.POSITION_CONTROL,
                                targetPositions=jointPoses,
                                forces=forces)




    p.stepSimulation()
    time.sleep(1./240.)


p.disconnect()
