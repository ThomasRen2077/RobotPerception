import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.animation as animation
from matplotlib import gridspec
from utils import *
from sort import SORTTracker
from pid import PIDController

def dynamic_camera_simulation(sensor_w, sensor_h, f, x_c, simulation_time, fps, output_file, plot_output_file):
    fig_3d = plt.figure(figsize=(15, 7))
    spec = gridspec.GridSpec(2, 2, width_ratios=[2, 1])

    ax_3d = fig_3d.add_subplot(spec[:, 0], projection='3d')
    ax_3d.set_xlim(0, 40)
    ax_3d.set_ylim(0, 40)
    ax_3d.set_zlim(0, 20)
    ax_3d.set_title("3D Camera FOV and Moving Objects")

    ax_psi_phi = fig_3d.add_subplot(spec[0, 1])
    ax_psi_phi.set_title("ψ (Pan) and φ (Tilt)")
    ax_psi_phi.set_xlabel("Time (s)")
    ax_psi_phi.set_ylabel("Radians")
    line_psi, = ax_psi_phi.plot([], [], label="ψ (Pan)", color="blue")
    line_phi, = ax_psi_phi.plot([], [], label="φ (Tilt)", color="orange")
    ax_psi_phi.legend()
    ax_psi_phi.grid(True)

    ax_psi_dot_phi_dot = fig_3d.add_subplot(spec[1, 1])
    ax_psi_dot_phi_dot.set_title("ψ̇ (Pan Velocity) and φ̇ (Tilt Velocity)")
    ax_psi_dot_phi_dot.set_xlabel("Time (s)")
    ax_psi_dot_phi_dot.set_ylabel("Radians/s")
    line_psi_dot, = ax_psi_dot_phi_dot.plot([], [], label="ψ̇ (Pan Velocity)", color="green")
    line_phi_dot, = ax_psi_dot_phi_dot.plot([], [], label="φ̇ (Tilt Velocity)", color="red")
    ax_psi_dot_phi_dot.legend()
    ax_psi_dot_phi_dot.grid(True)

    times = []
    psi_values = []
    phi_values = []
    psi_dot_values = []
    phi_dot_values = []

    plot_camera(ax_3d, x_c)

    camera_fov = CameraFOV(
        ax=ax_3d,
        psi=0, phi=0,
        psi_dot=0, phi_dot=0,
        f=f, sensor_w=sensor_w, sensor_h=sensor_h, x_c=x_c
    )

    # vehicle = LinearVehicle(ax, ax, x_offset=5, y_offset=5, vel_dir=[0.5, 0.2], color="C1")
    # vehicle2 = LinearVehicle(ax, ax, x_offset=10, y_offset=15, vel_dir=[-0.3, -0.1], color="C2")

    # Set up animal(s) in the scene
    animal1 = WildAnimal(ax_3d, x_init=5, y_init=5, color="C1")
    animal2 = WildAnimal(ax_3d, x_init=10, y_init=15, color="C4")

    # Initialize PID controllers
    pid_psi = PIDController(Kp=30, Ki=0.1, Kd=0.05, output_limits=(-1, 1))
    pid_phi = PIDController(Kp=30, Ki=0.1, Kd=0.05, output_limits=(-1, 1))

    # Non-PID controller
    # pid_psi = PIDController(Kp=30, Ki=0, Kd=0, output_limits=(-1, 1))
    # pid_phi = PIDController(Kp=30, Ki=0, Kd=0, output_limits=(-1, 1))

    sort_tracker = SORTTracker()
    tracked_artists = {}

    frames = int(simulation_time * fps)
    dt = 1 / fps

    state = {'fov_arrow': None}

    def update(frame):

        time = frame * dt  

        times.append(time)

        # vehicle.update_vehicle(time)
        # vehicle2.update_vehicle(time)
        animal1.update()
        animal2.update()

        targets = [animal1.position, animal2.position]
        distances = [np.linalg.norm(target - np.array(camera_fov.x_c)) for target in targets]
        closest_target = targets[np.argmin(distances)]

        target_vector = closest_target - np.array(camera_fov.x_c)

        desired_psi = np.arctan2(target_vector[1], target_vector[0])
        desired_phi = np.arctan2(target_vector[2], np.linalg.norm(target_vector[:2]))

        fov_direction = camera_fov.get_FOV_direction()
        current_psi = np.arctan2(fov_direction[1], fov_direction[0])
        current_phi = np.arctan2(fov_direction[2], np.linalg.norm(fov_direction[:2]))

        pid_psi.setpoint = desired_psi
        pid_phi.setpoint = desired_phi
        psi_dot = pid_psi.update(current_psi)
        phi_dot = pid_phi.update(current_phi)

        new_psi = camera_fov.psi + psi_dot * dt
        new_phi = camera_fov.phi + phi_dot * dt
        camera_fov.update_FOV(new_psi, new_phi)

        fov_direction *= 10

        if state['fov_arrow'] is not None:
            state['fov_arrow'].remove()

        state['fov_arrow'] = ax_3d.quiver(
            *camera_fov.x_c, *fov_direction, color="cyan", label="Camera FOV"
        )

        ax_3d.legend()

        psi_values.append(current_psi)
        phi_values.append(current_phi)
        psi_dot_values.append(psi_dot)
        phi_dot_values.append(phi_dot)

        line_psi.set_data(times, psi_values)
        line_phi.set_data(times, phi_values)
        ax_psi_phi.set_xlim(0, max(times))
        ax_psi_phi.set_ylim(-np.pi, np.pi)

        line_psi_dot.set_data(times, psi_dot_values)
        line_phi_dot.set_data(times, phi_dot_values)
        ax_psi_dot_phi_dot.set_xlim(0, max(times))
        ax_psi_dot_phi_dot.set_ylim(-1, 1)

        detections = [
            [animal1.position[0] - 0.5, animal1.position[1] - 0.5, animal1.position[0] + 0.5, animal1.position[1] + 0.5],
            [animal2.position[0] - 0.5, animal2.position[1] - 0.5, animal2.position[0] + 0.5, animal2.position[1] + 0.5]
        ]

        tracked_objects = sort_tracker.update(detections)

        for obj_id, bbox in tracked_objects:
            x_min, y_min, x_max, y_max = bbox
            cx = (x_min + x_max) / 2  
            cy = (y_min + y_max) / 2 

            ax_3d.plot3D(
                [x_min, x_max], [y_min, y_max], [0, 0], color="green"
            )

            if obj_id in tracked_artists:
                tracked_artists[obj_id]['text'].set_position((cx, cy))
                tracked_artists[obj_id]['text'].set_text(f"ID {obj_id}")
            else:
                text = ax_3d.text(cx, cy, 0, f"ID {obj_id}", fontsize=10, color="blue")
                line = ax_3d.plot3D([x_min, x_max], [y_min, y_max], [0, 0], color="green")[0]
                tracked_artists[obj_id] = {'text': text, 'line': line}

        for stale_id in list(tracked_artists.keys()):
            if stale_id not in [obj[0] for obj in tracked_objects]:
                tracked_artists[stale_id]['text'].remove()
                tracked_artists[stale_id]['line'].remove()
                del tracked_artists[stale_id]
            return []


    anim = animation.FuncAnimation(fig_3d, update, frames=frames, interval=1000 / fps, blit=False)
    anim.save(output_file, writer="pillow", fps=fps)
    plt.close(fig_3d)

    fig_plots, axs = plt.subplots(2, 2, figsize=(10, 12))

    axs = axs.flatten()

    axs[0].plot(times, psi_values, label="ψ (Pan)", color="blue")
    axs[0].set_title("Pan (ψ)")
    axs[0].set_xlabel("Time (s)")
    axs[0].set_ylabel("Radians")
    axs[0].grid(True)
    axs[0].legend()

    axs[1].plot(times, phi_values, label="φ (Tilt)", color="orange")
    axs[1].set_title("Tilt (φ)")
    axs[1].set_xlabel("Time (s)")
    axs[1].set_ylabel("Radians")
    axs[1].grid(True)
    axs[1].legend()

    axs[2].plot(times, psi_dot_values, label="ψ̇ (Pan Velocity)", color="green")
    axs[2].set_title("Pan Velocity (ψ̇)")
    axs[2].set_xlabel("Time (s)")
    axs[2].set_ylabel("Radians/s")
    axs[2].grid(True)
    axs[2].legend()

    axs[3].plot(times, phi_dot_values, label="φ̇ (Tilt Velocity)", color="red")
    axs[3].set_title("Tilt Velocity (φ̇)")
    axs[3].set_xlabel("Time (s)")
    axs[3].set_ylabel("Radians/s")
    axs[3].grid(True)
    axs[3].legend()

    fig_plots.tight_layout()
    fig_plots.savefig(plot_output_file)
    plt.close(fig_plots)

    print(f"Plots saved to {plot_output_file}")

    return anim

if __name__ == "__main__":
    anim = dynamic_camera_simulation(
        sensor_w=20, sensor_h=20, f=50, x_c=[0, 20, 15],
        simulation_time=10, fps=30,
        output_file="sim.gif",
        plot_output_file="angles.png"
    )
