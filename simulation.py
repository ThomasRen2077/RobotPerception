import numpy as np
import matplotlib.pyplot as plt

from desired_angle import desired_angle
from controller import controller
from kinematic_cam import kinematic_cam
from measurement_cam import measurement_cam

T0 = 0
Tf = 50
Nstep = 500
delt = (Tf - T0) / Nstep

# Target initial positions and velocities
xT = np.array([2.0, 2.0])
xTdot = np.array([0.05, 0.1])

xT2 = np.array([3.0, 3.0])
xTdot2 = np.array([-0.05, -0.1])

# Camera state and control
s = np.array([2.3562, 2.5261, 0, 0])
u = np.array([0, 0])
lambda_ = 4
lambda_min = 2
lambda_max = 6
adjustment_rate = 0.1
error_threshold = 0.5

# Records for plotting
x_rec, z_rec, x_rec2, z_rec2 = [], [], [], []
s_rec, u_rec = [], []

# Simulation loop
for t in np.linspace(T0, Tf, Nstep):
    z, z2 = measurement_cam(s, xT, xTdot, xT2, xTdot2, lambda_)
    
    if np.any(np.abs(z) > error_threshold):
        if z[0] > error_threshold:
            lambda_ = min(lambda_max, lambda_ + adjustment_rate)
        else:
            lambda_ = max(lambda_min, lambda_ - adjustment_rate)
    
    u = controller(z, s, delt, u, lambda_)
    x_rec.append(xT.copy())
    z_rec.append(z.copy())
    x_rec2.append(xT2.copy())
    z_rec2.append(z2.copy())
    s_rec.append(s.copy())
    u_rec.append(u.copy())

    s = kinematic_cam(s, u, delt)

    # Random walks for the cats
    theta, theta2 = np.random.rand() * 2 * np.pi, np.random.rand() * 2 * np.pi
    xT += 0.6 * np.array([np.cos(theta), np.sin(theta)]) * delt
    xT2 += 0.6 * np.array([np.cos(theta2), np.sin(theta2)]) * delt

# Plotting target trajectories in the inertial frame
plt.figure(figsize=(10, 8))
plt.subplot(3, 2, 1)
plt.plot(np.array(x_rec)[:, 0], np.array(x_rec)[:, 1], 'b', linewidth=2, label='Cat 1')
plt.plot(np.array(x_rec2)[:, 0], np.array(x_rec2)[:, 1], 'r', linewidth=2, label='Cat 2')
plt.grid(True)
plt.title('Target Trajectories in Inertial Frame')
plt.xlabel('X (m)')
plt.ylabel('Y (m)')
plt.legend()


# Plotting targets in the camera frame
plt.subplot(3, 2, 2)
plt.plot(np.array(z_rec)[:, 0], np.array(z_rec)[:, 1], 'b', linewidth=2, label='Projection Cat 1')
plt.plot(np.array(z_rec2)[:, 0], np.array(z_rec2)[:, 1], 'r', linewidth=2, label='Projection Cat 2')
plt.grid(True)
plt.title('Targets in Camera Frame')
plt.xlabel('p_x (pixels)')
plt.ylabel('p_y (pixels)')
plt.legend()

time = np.linspace(T0, Tf, Nstep)
# Plotting camera angles over time
plt.subplot(3, 2, 3)
plt.plot(time, np.array(s_rec)[:, 0], 'b-', label='Pan Angle')
plt.title('Camera Pan Angle Over Time')
plt.ylabel('Pan Angle (degrees)')
plt.xlabel('Time (s)')
plt.grid(True)

plt.subplot(3, 2, 4)
plt.plot(time, np.array(s_rec)[:, 1], 'r-', label='Tilt Angle')
plt.title('Camera Tilt Angle Over Time')
plt.ylabel('Tilt Angle (degrees)')
plt.xlabel('Time (s)')
plt.grid(True)
plt.tight_layout()

# Plotting control signals
plt.subplot(3, 2, 5)
plt.plot(time, np.array(u_rec)[:, 0], 'b-', label='Control Signal for Pan')
plt.title('Control Voltage Input for Pan')
plt.ylabel('Voltage (V)')
plt.xlabel('Time (s)')
plt.grid(True)

plt.subplot(3, 2, 6)
plt.plot(time, np.array(u_rec)[:, 1], 'r-', label='Control Signal for Tilt')
plt.title('Control Voltage Input for Tilt')
plt.ylabel('Voltage (V)')
plt.xlabel('Time (s)')
plt.grid(True)
plt.tight_layout()

plt.show()
