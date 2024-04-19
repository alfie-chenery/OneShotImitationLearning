import pybullet as p
import time
import pybullet_data
import pickle
import os
import numpy as np

p.connect(p.GUI)
p.resetSimulation()
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0,0,-10)

planeId = p.loadURDF("plane.urdf")
armId = p.loadURDF("franka_panda/panda.urdf", [0,0,0], [0,0,0,1], useFixedBase=True)
tableId = p.loadURDF("table/table.urdf", [0.6,0,-0.2], p.getQuaternionFromEuler([0,0,np.pi/2]))
numJoints = p.getNumJoints(armId)
end_effector_id = numJoints - 1

path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trace.pkl")
with open(path, 'rb') as f:
    trace = pickle.load(f)

interpolation_steps = 100 #steps between each trace keyframe

for keyFrame in range(len(trace)):

    prev_positions = [state[0] for state in p.getJointStates(armId, range(numJoints))]
    desired_positions = trace[keyFrame]

    for i in range(interpolation_steps):
        alpha = (i+1) / interpolation_steps
        interpolated_position = [(1 - alpha) * prev + alpha * desired for prev, desired in zip(prev_positions, desired_positions)]
        forces = [500.0] * len(interpolated_position)

        p.setJointMotorControlArray(armId, 
                                    range(len(interpolated_position)),
                                    p.POSITION_CONTROL,
                                    targetPositions=interpolated_position,
                                    forces=forces)


        p.stepSimulation()
        time.sleep(1./240.)


p.disconnect()
