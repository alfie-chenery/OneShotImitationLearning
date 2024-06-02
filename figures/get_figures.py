import environment
import pybullet as p
import numpy as np
import os
import matplotlib.pyplot as plt
import cv2


dir_path = os.path.dirname(os.path.realpath(__file__))
env = environment.FrankaArmEnvironment()

print(env.robotGetEefPosition())
p.resetBasePositionAndOrientation(env.objectId, [0.56,0,0.45], p.getQuaternionFromEuler([0,0,0]))

env.robotSaveCameraSnapshot("figure", dir_path)



