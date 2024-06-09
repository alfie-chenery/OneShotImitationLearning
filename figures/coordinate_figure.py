import pybullet as p
import pybullet_data
import numpy as np
import time
from PIL import Image
import os

p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setRealTimeSimulation(0)
p.setGravity(0, 0, 0)
# p.configureDebugVisualizer(p.COV_ENABLE_GUI,0)
p.resetDebugVisualizerCamera(0.7, 15.0, -30.0, [0.15, 0.15, 0.1]) #yaw 15 for fig1, 180 for fig2

def drawLocalFrame(objectId, thickness, dotted=False):
    pos, orn = p.getBasePositionAndOrientation(objectId)
    orn = np.array(p.getMatrixFromQuaternion(orn)).reshape((3,3))

    localX = orn.dot([1,0,0])
    localY = orn.dot([0,1,0])
    localZ = orn.dot([0,0,1])
    red = [1,0,0]
    green = [0,1,0]
    blue = [0,0,1]

    for line, colour in zip([localX, localY, localZ], [red, green, blue]):

        step = 0.02 if dotted else 0.5
        endpoints = np.arange(0, 0.5, step).tolist() + [0.5]
        pairs = zip(endpoints[::2], endpoints[1::2])

        for i, j in pairs:
            start = pos + i * line
            end = pos + j * line
            p.addUserDebugLine(start, end, colour, thickness)

def drawArc(cx, cy, radius, z, thetaStart, thetaEnd, steps=16, arrow=False):
    """
    Assumes drawing in the XY plane at altitude z
    ThetaStart and ThetaEnd in radians
    """
    prevX = None
    prevY = None
    thetas = np.linspace(thetaStart, thetaEnd, steps, endpoint=True).tolist()
    pairs = zip(thetas, thetas[1:])
    for theta, phi in pairs:
        x1 = cx + radius * np.cos(theta)
        y1 = cy + radius * np.sin(theta)
        x2 = cx + radius * np.cos(phi)
        y2 = cy + radius * np.sin(phi)

        p.addUserDebugLine([x1,y1,z], [x2,y2,z], [0,0,0], 1)


    if arrow:
        dir = np.array([x2-x1, y2-y1]) #direction the arc travels in at the last segment
        dir = dir / np.linalg.norm(dir)
        normal = np.array([-dir[1], dir[0]]) #90 degrees to dir
        
        head = np.array([x2, y2])
        arrowLength = 0.02

        corner1 = head - (dir + normal / 2) * arrowLength
        corner2 = head - (dir - normal / 2) * arrowLength

        p.addUserDebugLine([head[0], head[1], z], [corner1[0], corner1[1], z], [0,0,0], 1)
        p.addUserDebugLine([head[0], head[1], z], [corner2[0], corner2[1], z], [0,0,0], 1)



dir_path = os.path.dirname(os.path.realpath(__file__))

# planeId = p.loadURDF("plane.urdf")
objectId = p.loadURDF("block.urdf", [0.1, 0.1, 0.1], [0,0,0,1])
object2Id = p.loadURDF("block.urdf", [0.1, 0.1, 0.1], p.getQuaternionFromEuler([0,0,np.pi/4]))

p.changeVisualShape(objectId, -1, rgbaColor=[1, 1, 0, 0.5])
p.changeVisualShape(object2Id, -1, rgbaColor=[1, 1, 0, 0.5])


drawLocalFrame(objectId, 2, dotted=True)    #object 1 is the initial frame, dotted line, where it used to be
drawLocalFrame(object2Id, 2, dotted=False)  # object 2 is the rotated orientation
drawArc(0.1, 0.1, 0.2, 0.1, 0, np.pi/4, 16, arrow=True)
drawArc(0.1, 0.1, 0.2, 0.1, np.pi/2, 3*np.pi/4, 16, arrow=True)


time.sleep(5)
while True:
    pass
    # _, _, _, _, _, _, _, _, cameraYaw, cameraPitch, cameraDist, cameraTarget = p.getDebugVisualizerCamera()
    # cameraYaw += 0.0001
    # p.resetDebugVisualizerCamera(cameraDist, cameraYaw, cameraPitch, cameraTarget)