import environment
import pickle
import os

env = environment.FrankaArmEnvironment()

path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trace.pkl")
with open(path, 'rb') as f:
    trace = pickle.load(f)

eefMode = False # True, set robot angles, False set root eef by inverse kinematics

for keyFrame in range(len(trace)):
    if eefMode:
        desired_pos, desired_orn = trace[keyFrame][0]
        env.robotSetEefPosition(desired_pos, desired_orn)
    else:
        desired_angles = trace[keyFrame][1]
        env.robotSetJointAngles(desired_angles)

env.closeEnv()
