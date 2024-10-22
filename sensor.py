import numpy as np
import matplotlib.pyplot as plt

def measurement_cam(s, xT, xTdot, xT2, xTdot2, lambda_):
    """Simulate camera measurements of targets in the camera's coordinate frame."""
    xC = np.array([0, 0, 4])  # Camera position in frame F
    psi, phi, psiDot, phiDot = s
    R_phi = np.array([[1, 0, 0], [0, np.cos(phi), np.sin(phi)], [0, -np.sin(phi), np.cos(phi)]])
    R_psi = np.array([[np.cos(psi), np.sin(psi), 0], [-np.sin(psi), np.cos(psi), 0], [0, 0, 1]])
    
    qT = R_phi @ R_psi @ (np.hstack([xT, 0]) - xC)
    qT2 = R_phi @ R_psi @ (np.hstack([xT2, 0]) - xC)
    
    pT = lambda_ * (qT[:2] / qT[2])
    pT2 = lambda_ * (qT2[:2] / qT2[2])
    return np.hstack([pT, pT2])

def controller(z, s, delt, u_prev, lambda_):
    """Control the camera to track the targets."""
    b1 = b2 = 100 * np.pi / 180  # Control sensitivity
    z1, z2 = z[:2], z[2:]
    PandT1 = desired_angle(z1, s, lambda_)
    PandT2 = desired_angle(z2, s, lambda_)
    
    delpsi = (PandT1[0] + PandT2[0]) / 2 - s[0]
    delphi = (PandT1[1] + PandT2[1]) / 2 - s[1]
    
    u_now = np.array([(delpsi / delt - s[2]) / b1, (delphi / delt - s[3]) / b2])
    dudt = (u_now - u_prev) / delt
    kp, kd = 0.3, 0.03  # Proportional and derivative gains
    u = kp * u_now + kd * dudt
    return np.clip(u, -1, 1)

def desired_angle(z, s, lambda_):
    """Calculate the desired camera angles to track a target."""
    px, py = z
    psi, phi = s[:2]
    R_phi = np.array([[1, 0, 0], [0, np.cos(phi), np.sin(phi)], [0, -np.sin(phi), np.cos(phi)]])
    R_psi = np.array([[np.cos(psi), np.sin(psi), 0], [-np.sin(psi), np.cos(psi), 0], [0, 0, 1]])
    P = np.array([px, py, lambda_])
    xT_recon = -lambda_ / (R_phi[2] @ R_psi @ P) * R_psi.T @ R_phi.T @ np.array([px, py, lambda_]) - np.array([0, 0, -lambda_])
    pan = np.pi / 2 + np.arctan2(xT_recon[1], xT_recon[0])
    tilt = np.pi / 2 + np.arctan2(4, np.linalg.norm(xT_recon[:2]))
    return np.array([pan, tilt])

def kinematic_cam(s, u, delt):
    """Update the camera state based on control inputs."""
    psiDot_m = phiDot_m = 100 * np.pi / 180
    A = np.array([[1, 0, delt, 0], [0, 1, 0, delt], [0, 0, 1, 0], [0, 0, 0, 1]])
    B = np.array([[0, 0], [0, 0], [100 * np.pi/180, 0], [0, 100 * np.pi/180]])
    s_next = A @ s + B @ u
    s_next = np.clip(s_next, [0, 0, -psiDot_m, -phiDot_m], [2*np.pi, np.pi, psiDot_m, phiDot_m])
    return s_next

# Simulation parameters
T0 = 0
Tf = 50
Nstep = 500
delt = (Tf - T0) / Nstep

# Initialize state variables
s = np.array([2.3562, 2.5261, 0, 0])
u = np.array([0, 0])
lambda_ = 4
xT = np.array([2.0, 2.0])
xTdot = np.array([0.05, 0.1])
xT2 = np.array([3.0, 3.0])
xTdot2 = np.array([-0.05, -0.1])

# Record for plotting
trajectories = []
angles = []

# Simulation loop
for t in np.arange(T0, Tf, delt):
    z = measurement_cam(s, xT, xTdot, xT2, xTdot2, lambda_)
    u = controller(z, s, delt, u, lambda_)
    s = kinematic_cam(s, u, delt)
    xT += xTdot * delt
    xT2 += xTdot2 * delt
    trajectories.append(np.hstack((xT, xT2)))
    angles.append(s[:2])

# Plotting
trajectories = np.array(trajectories)
angles = np.array(angles)
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.plot(trajectories[:, 0], trajectories[:, 1], label='Cat 1')
plt.plot(trajectories[:, 2], trajectories[:, 3], label='Cat 2')
plt.title('Position Trajectories')
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.legend()

plt.subplot(122)
plt.plot(angles[:, 0], label='Pan Angle')
plt.plot(angles[:, 1], label='Tilt Angle')
plt.title('Camera Angles')
plt.xlabel('Time Steps')
plt.ylabel('Angle (radians)')
plt.legend()
plt.tight_layout()
plt.show()
