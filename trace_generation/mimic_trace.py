import environment
import pickle
import os

dir_path = os.path.dirname(os.path.abspath(__file__))

env = environment.FrankaArmEnvironment(videoLogging=True, out_dir=os.path.join(dir_path, "out"))

with open(os.path.join(dir_path, "trace.pkl"), 'rb') as f:
    trace = pickle.load(f)

eefMode = False # True, set robot angles, False set robot eef by inverse kinematics

for keyFrame in range(len(trace)):
    if eefMode:
        desired_pos, desired_orn = trace[keyFrame][0]
        env.robotSetEefPosition(desired_pos, desired_orn)
    else:
        desired_angles = trace[keyFrame][1]
        env.robotSetJointAngles(desired_angles)

env.closeEnv()
