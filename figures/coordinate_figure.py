import pybullet as p
import pybullet_data
import numpy as np
import time

p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setRealTimeSimulation(0)
p.setGravity(0, 0, 0)
p.resetDebugVisualizerCamera(0.5, 45.0, -30.0, [0.2, 0.2, 0.2])

def drawLocalFrame(objectId, thickness):
    pos, orn = p.getBasePositionAndOrientation(objectId)
    orn = np.array(p.getMatrixFromQuaternion(orn)).reshape((3,3))

    localX = orn.dot([1,0,0])
    localY = orn.dot([0,1,0])
    localZ = orn.dot([0,0,1])

    p.addUserDebugLine(pos, pos + 0.5 * localX, [1,0,0], thickness)
    p.addUserDebugLine(pos, pos + 0.5 * localY, [0,1,0], thickness)
    p.addUserDebugLine(pos, pos + 0.5 * localZ, [0,0,1], thickness)


# planeId = p.loadURDF("plane.urdf")
objectId = p.loadURDF("block.urdf", [0.2, 0.2, 0.2], [0,0,0,1])
object2Id = p.loadURDF("block.urdf", [0.2, 0.2, 0.2], p.getQuaternionFromEuler([0,-np.pi/4,0]))

p.changeVisualShape(objectId, -1, rgbaColor=[1, 1, 0, 0.5])
p.changeVisualShape(object2Id, -1, rgbaColor=[1, 1, 0, 0.5])


drawLocalFrame(objectId, 1)
drawLocalFrame(object2Id, 2)

while True:
    time.sleep(1./240.)