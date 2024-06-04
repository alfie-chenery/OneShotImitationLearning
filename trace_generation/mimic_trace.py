import environment
import pickle
import os

dir_path = os.path.dirname(os.path.abspath(__file__))

env = environment.FrankaArmEnvironment(videoLogging=True, out_dir=os.path.join(dir_path, "out"))

with open(os.path.join(dir_path, "demonstration.pkl"), 'rb') as f:
    trace = pickle.load(f)

eefMode = True # True, set robot eef by inverse kinematics, False set robot joint angles

for keyFrame in range(len(trace)):
    if eefMode:
        desired_pos, desired_orn, gripper_closed = trace[keyFrame]
        env.robotSetEefPosition(desired_pos, desired_orn)
        env.robotCloseGripper() if gripper_closed else env.robotOpenGripper()
    else:
        desired_angles = trace[keyFrame]
        env.robotSetJointAngles(desired_angles)

env.closeEnv()
