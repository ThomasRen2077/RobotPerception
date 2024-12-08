import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# Import CameraFOV class from utils.py and PIDController
from utils import CameraFOV
from pid import PIDController  # Make sure pid_controller.py is in the same directory

# Simulated Moving Targets
def moving_target_trajectory_1(frame):
    """Generate trajectory for Target 1."""
    x = 10 + frame * 0.5
    y = 10 + np.sin(frame * 0.1) * 5
    z = 10 + np.cos(frame * 0.1) * 3
    return np.array([x, y, z])

def moving_target_trajectory_2(frame):
    """Generate trajectory for Target 2."""
    x = 40 - frame * 0.3
    y = 40 - np.cos(frame * 0.1) * 5
    z = 10 + np.sin(frame * 0.1) * 3
    return np.array([x, y, z])

# Initialize variables to store the animation
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.set_xlim(0, 50)
ax.set_ylim(0, 50)
ax.set_zlim(0, 20)

# Initialize CameraFOV
camera = CameraFOV(
    ax=ax, 
    x_c=[20, 20, 10], 
    psi=0, 
    phi=np.pi / 2, 
    psi_dot=0, 
    phi_dot=0, 
    f=50, 
    sensor_w=10, 
    sensor_h=10
)

# Initialize PID controllers for pan (ψ) and tilt (φ)
pid_psi = PIDController(Kp=0.6, Ki=0.1, Kd=0.05, output_limits=(-2, 2))
pid_phi = PIDController(Kp=0.6, Ki=0.1, Kd=0.05, output_limits=(-2, 2))

current_target = 0  # Start by tracking Target 1
hysteresis_threshold = 2.0  # Threshold for switching targets (e.g., 2 units)

def update(frame):
    global current_target

    # Simulate positions of two targets
    target_1 = moving_target_trajectory_1(frame)
    target_2 = moving_target_trajectory_2(frame)

    # List of targets
    targets = [target_1, target_2]

    # Compute distances from the camera to each target
    distances = [np.linalg.norm(target - camera.x_c) for target in targets]

    # Check if we need to switch the target
    if current_target == 0:
        # Currently tracking Target 1
        if distances[1] < distances[0] - hysteresis_threshold:
            current_target = 1  # Switch to Target 2
    else:
        # Currently tracking Target 2
        if distances[0] < distances[1] - hysteresis_threshold:
            current_target = 0  # Switch to Target 1

    # Select the current target
    closest_target = targets[current_target]

    # Compute the vector from the camera to the closest target
    target_vector = closest_target - np.array(camera.x_c)

    # Calculate desired pan (ψ) and tilt (φ) angles
    desired_psi = np.arctan2(target_vector[1], target_vector[0])
    desired_phi = np.arctan2(target_vector[2], np.linalg.norm(target_vector[:2]))

    # Update camera's FOV using PID controllers
    pid_psi.setpoint = desired_psi
    pid_phi.setpoint = desired_phi

    voltage_psi = pid_psi.update(camera.psi)  # Time-scaled voltage
    voltage_phi = pid_phi.update(camera.phi)  # Time-scaled voltage

    camera.psi += voltage_psi
    camera.phi += voltage_phi

    # Compute FOV direction based on camera's psi and phi
    fov_direction = np.array([
        np.cos(camera.phi) * np.cos(camera.psi),
        np.cos(camera.phi) * np.sin(camera.psi),
        np.sin(camera.phi)
    ])

    # Scale the FOV direction for visualization
    fov_length = 10
    fov_vector = fov_direction * fov_length

    # Clear the plot and redraw the updated positions and directions
    ax.clear()
    ax.set_xlim(0, 50)
    ax.set_ylim(0, 50)
    ax.set_zlim(0, 20)
    ax.scatter(*target_1, color="red", label="Target 1")
    ax.scatter(*target_2, color="blue", label="Target 2")
    ax.scatter(*closest_target, color="purple", label="Tracked Target")
    ax.scatter(*camera.x_c, color="green", label="Camera")
    ax.quiver(
        *camera.x_c,  # Camera position
        *fov_vector,  # Scaled FOV direction
        color="cyan",
        label="Camera FOV Direction",
        length=10
    )
    ax.legend()

# Create animation
anim = FuncAnimation(fig, update, frames=500, interval=100)

# Save animation as a GIF
anim.save("test_tracking.gif", writer="pillow", fps=20)

print("Animation saved as camera_tracking_pid_controller.gif")
