import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.animation as animation
from utils import *
from sort import SORTTracker

def dynamic_camera_simulation(sensor_w, sensor_h, f, x_c, psi_func, phi_func, simulation_time, fps, output_file):
    """
    Simulate a dynamic camera with objects moving in its field of view and save the animation as a GIF.
    """
    fig = plt.figure(figsize=(12, 7))
    ax = fig.add_subplot(111, projection='3d')

    ax.set_xlim(0, 40)  
    ax.set_ylim(0, 40)  
    ax.set_zlim(0, 20) 

    plot_camera(ax, x_c)

    # Initialize dynamic camera
    camera_fov = CameraFOV(
        ax=ax,
        psi=psi_func(0), phi=phi_func(0),
        psi_dot=0, phi_dot=0,
        f=f, sensor_w=sensor_w, sensor_h=sensor_h, x_c=x_c
    )

    # Set up vehicle(s) in the scene
    # vehicle = LinearVehicle(ax, ax, x_offset=5, y_offset=5, vel_dir=[0.5, 0.2], color="C1")
    # vehicle2 = LinearVehicle(ax, ax, x_offset=10, y_offset=15, vel_dir=[-0.3, -0.1], color="C2")

    # Set up animal(s) in the scene
    animal1 = WildAnimal(ax, x_init=5, y_init=5, color="C1")
    animal2 = WildAnimal(ax, x_init=10, y_init=15, color="C4")

    sort_tracker = SORTTracker()
    tracked_artists = {}

    # Set up simulation parameters
    frames = int(simulation_time * fps)
    dt = 1 / fps

    def update(frame):
        time = frame * dt  

        # Update camera FOV
        psi = psi_func(time)  
        phi = phi_func(time)  
        camera_fov.update_FOV(psi, phi)

        # Update vehicle movements
        # vehicle.update_vehicle(time)
        # vehicle2.update_vehicle(time)
        animal1.update()
        animal2.update()

        # Gather detections (bounding boxes)
        detections = [
            [animal1.position[0] - 0.5, animal1.position[1] - 0.5, animal1.position[0] + 0.5, animal1.position[1] + 0.5],
            [animal2.position[0] - 0.5, animal2.position[1] - 0.5, animal2.position[0] + 0.5, animal2.position[1] + 0.5]
        ]

        # Update SORT tracker
        tracked_objects = sort_tracker.update(detections)

        # Reuse or update existing artists
        for obj_id, bbox in tracked_objects:
            x_min, y_min, x_max, y_max = bbox
            cx = (x_min + x_max) / 2  # Center X
            cy = (y_min + y_max) / 2  # Center Y

            ax.plot3D(
                [x_min, x_max], [y_min, y_max], [0, 0], color="green"
            )

            # Update or create text label
            if obj_id in tracked_artists:
                tracked_artists[obj_id]['text'].set_position((cx, cy))
                tracked_artists[obj_id]['text'].set_text(f"ID {obj_id}")
            else:
                # Create a new label
                text = ax.text(cx, cy, 0, f"ID {obj_id}", fontsize=10, color="blue")
                line = ax.plot3D([x_min, x_max], [y_min, y_max], [0, 0], color="green")[0]
                tracked_artists[obj_id] = {'text': text, 'line': line}

        # Remove any stale artists
        for stale_id in list(tracked_artists.keys()):
            if stale_id not in [obj[0] for obj in tracked_objects]:
                tracked_artists[stale_id]['text'].remove()
                tracked_artists[stale_id]['line'].remove()
                del tracked_artists[stale_id]
            return []

    anim = animation.FuncAnimation(fig, update, frames=frames, interval=1000 / fps, blit=False)
    anim.save(output_file, writer="pillow", fps=fps)
    plt.show()

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
        sensor_w=20, sensor_h=20, f=50, x_c=[0, 20, 15],
        psi_func=psi_func, phi_func=phi_func,
        simulation_time=10, fps=30,
        output_file="sim.gif"
    )
