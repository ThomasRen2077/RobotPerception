import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.animation as animation
from my_utils import *

# Use the provided helper functions and classes (CameraFOV, LinearVehicle) here.

def dynamic_camera_simulation(sensor_w, sensor_h, f, x_c, psi_func, phi_func, simulation_time, fps, output_file):
    """
    Simulate a dynamic camera with objects moving in its field of view and save the animation as a GIF.
    """
    fig = plt.figure(figsize=(12, 7))
    ax = fig.add_subplot(111, projection='3d')

    ax.set_xlim(0, 40)  
    ax.set_ylim(0, 40)  
    ax.set_zlim(0, 20) 

    # Initialize dynamic camera
    camera_fov = CameraFOV(
        ax=ax,
        psi=psi_func(0), phi=phi_func(0),
        psi_dot=0, phi_dot=0,
        f=f, sensor_w=sensor_w, sensor_h=sensor_h, x_c=x_c
    )

    # Set up vehicle(s) in the scene
    vehicle = LinearVehicle(ax, ax, x_offset=5, y_offset=5, vel_dir=[0.5, 0.2], color="C1")
    vehicle2 = LinearVehicle(ax, ax, x_offset=10, y_offset=15, vel_dir=[-0.3, -0.1], color="C2")

    # Set up simulation parameters
    frames = int(simulation_time * fps)
    dt = 1 / fps

    def update(frame):
        time = frame * dt  # Calculate time

        # Update camera FOV
        psi = psi_func(time)  # Dynamic pan angle
        phi = phi_func(time)  # Dynamic tilt angle
        camera_fov.update_FOV(psi, phi)

        # Update vehicle movements
        vehicle.update_vehicle(time)
        vehicle2.update_vehicle(time)

        return []

    # Create animation
    anim = animation.FuncAnimation(fig, update, frames=frames, interval=1000 / fps, blit=False)

    # Save the animation as a GIF
    anim.save(output_file, writer="pillow", fps=fps)

    # Show the plot
    plt.show()

    # Return the animation object
    return anim

# Example dynamic camera movement functions
def psi_func(t):
    """Pan angle: Oscillates between -π/4 and π/4."""
    return np.pi / 4 * np.sin(2 * np.pi * t / 5)  # 5-second oscillation

def phi_func(t):
    """Tilt angle: Oscillates between π/6 and π/3."""
    return np.pi / 6 + (np.pi / 6) * np.sin(2 * np.pi * t / 10)  # 10-second oscillation

# Run the simulation and save the output
if __name__ == "__main__":
    anim = dynamic_camera_simulation(
        sensor_w=10, sensor_h=10, f=50, x_c=[20, 20, 10],
        psi_func=psi_func, phi_func=phi_func,
        simulation_time=10, fps=30,
        output_file="dynamic_camera_simulation.gif"
    )
