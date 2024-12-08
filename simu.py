import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.animation import FuncAnimation
from my_utils import *

# Initialize the 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

# Plot the camera
x_c = np.array([0, 0, 0])  # Camera position
plot_camera(ax, x_c)

# Initialize the CameraFOV object
camera_fov = CameraFOV(
    ax=ax, psi=0, phi=0, psi_dot=0.1, phi_dot=0.1, 
    f=10, sensor_w=4, sensor_h=3, x_c=x_c
)

# Animation update function
def update(frame):
    psi = 20 * np.sin(frame / 50)  # Simulated pan angle
    phi = 10 * np.cos(frame / 50)  # Simulated tilt angle
    camera_fov.update_FOV(psi, phi)
    return camera_fov.fov_corners, camera_fov.fov_collection

# Run the animation
ani = FuncAnimation(fig, update, frames=500, interval=50, blit=False)
plt.show()
