import matplotlib.pyplot as plt
import numpy as np
import pybullet as p
import pybullet_data as pd

p.connect(p.GUI)
p.setAdditionalSearchPath(pd.getDataPath())
p.loadURDF('plane.urdf')
p.loadURDF('cube_small.urdf', basePosition=[0.0, 0.0, 0.025])

width = 128
height = 128

fov = 60
aspect = width / height
near = 0.02
far = 1

view_matrix = p.computeViewMatrix([0, 0, 0.5], [0, 0, 0], [1, 0, 0])
projection_matrix = p.computeProjectionMatrixFOV(fov, aspect, near, far)

# Get depth values using the OpenGL renderer
images = p.getCameraImage(width, height, view_matrix, projection_matrix, renderer=p.ER_BULLET_HARDWARE_OPENGL)
depth_buffer_opengl = np.reshape(images[3], [width, height])
depth_opengl = far * near / (far - (far - near) * depth_buffer_opengl)

# Get depth values using Tiny renderer
images = p.getCameraImage(width, height, view_matrix, projection_matrix, renderer=p.ER_TINY_RENDERER)
depth_buffer_tiny = np.reshape(images[3], [width, height])
depth_tiny = far * near / (far - (far - near) * depth_buffer_tiny)

p.disconnect()

# Plot both images - should show depth values of 0.45 over the cube and 0.5 over the plane
plt.subplot(1, 2, 1)
plt.imshow(depth_opengl, cmap='gray', vmin=0, vmax=1)
plt.title('OpenGL Renderer')

plt.subplot(1, 2, 2)
plt.imshow(depth_tiny, cmap='gray', vmin=0, vmax=1)
plt.title('Tiny Renderer')

plt.show()