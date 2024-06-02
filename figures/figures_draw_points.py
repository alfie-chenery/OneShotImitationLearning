# import environment
# import pybullet as p
import numpy as np
import os
import matplotlib.pyplot as plt
import cv2


dir_path = os.path.dirname(os.path.realpath(__file__))
# env = environment.FrankaArmEnvironment()

# print(env.robotGetEefPosition())
# p.resetBasePositionAndOrientation(env.objectId, [0.56,0,0.45], p.getQuaternionFromEuler([0,0,0]))

# env.robotSaveCameraSnapshot("figure", dir_path + "\\out")


img1 = rgb = cv2.imread(dir_path + f"\\figure1b.jpg", cv2.IMREAD_COLOR)
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
img2 = rgb = cv2.imread(dir_path + f"\\figure1.jpg", cv2.IMREAD_COLOR)
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

# Manual keypoints coordinates
img1_keypoints = [(152, 328), (290, 328), (152, 466), (290, 466), (309, 334), (309, 462)]
img2_keypoints = [(274, 397), (372, 299), (372, 495), (470, 397)]

# img1_avg = tuple(map(lambda x: sum(x) / len(img1_keypoints), zip(*img1_keypoints)))
# img2_avg = tuple(map(lambda x: sum(x) / len(img2_keypoints), zip(*img2_keypoints)))
# img1_keypoints = [img1_avg]
# img2_keypoints = [img2_avg]

# Convert coordinates to KeyPoint objects
kp1 = [cv2.KeyPoint(x, y, 1) for (x, y) in img1_keypoints]
kp2 = [cv2.KeyPoint(x, y, 1) for (x, y) in img2_keypoints]

# Create DMatch objects
shorter_length = min(len(kp1), len(kp2))
matches = [cv2.DMatch(i, i, 0) for i in range(shorter_length)]

# Add a dividing line
width = 2 # pixels
divider = np.full((img1.shape[0], width, 3), 0, dtype=np.uint8)
img1_divider = np.hstack((img1, divider))

# Draw matches
matchImg = cv2.drawMatches(img1_divider, kp1, img2, kp2, matches, None, matchColor=[0,255,0], singlePointColor=[255,0,0])# , flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# Save and display image
plt.figure(figsize = (12,8))
plt.imshow(matchImg)
plt.savefig(dir_path + "\\fig.png", bbox_inches="tight")
plt.show()


