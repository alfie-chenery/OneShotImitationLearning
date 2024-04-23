from main_code import environment
import pickle
import os

env = environment.FrankaArmEnvironment()

path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trace.pkl")
with open(path, 'rb') as f:
    trace = pickle.load(f)

for keyFrame in range(len(trace)):
    desired_positions = trace[keyFrame]
    env.robotSetJointAngles(desired_positions)

env.closeEnv()
