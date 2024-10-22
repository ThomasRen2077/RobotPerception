import numpy as np
import matplotlib.pyplot as plt

# Constants
T0, Tf, Nstep = 0, 50, 500
delt = (Tf - T0) / Nstep
lambda_ = 4  # Focal length
max_psi_dot = max_phi_dot = 100 * np.pi / 180  # Max angular velocities
b1 = b2 = 100 * np.pi / 180  # Motor coefficients

def simulate_target_movement(initial_pos, delt, steps):
    """Generate random movements for target."""
    trajectory = [initial_pos]
    for _ in range(1, steps):
        movement = np.random.normal(loc=0.0, scale=0.1, size=2)
        new_position = trajectory[-1] + movement
        trajectory.append(new_position)
    return np.array(trajectory)

def camera_projection(xT, s):
    """Project target position onto virtual image plane."""
    psi, phi = s[:2]
    R_psi = np.array([[np.cos(psi), -np.sin(psi), 0],
                      [np.sin(psi), np.cos(psi), 0],
                      [0, 0, 1]])
    R_phi = np.array([[1, 0, 0],
                      [0, np.cos(phi), np.sin(phi)],
                      [0, -np.sin(phi), np.cos(phi)]])
    xC = np.array([0, 0, lambda_])  # Camera coordinates
    qT = np.dot(R_phi, np.dot(R_psi, np.hstack([xT, [0]]) - xC))
    if qT[2] == 0:
        qT[2] = 1e-6  # Avoid division by zero
    pT = lambda_ * (qT[:2] / qT[2])  # Projection on image plane
    return pT

def update_camera(s, target, delt):
    """Update camera angles to track the target."""
    psi_dot = max_psi_dot * np.clip(target[0], -1, 1)
    phi_dot = max_phi_dot * np.clip(target[1], -1, 1)
    new_psi = s[0] + psi_dot * delt
    new_phi = s[1] + phi_dot * delt
    return np.array([new_psi, new_phi])

# Initial positions and camera state
target1_pos = np.array([2, 2])
target2_pos = np.array([3, 3])
camera_state = np.array([0, np.pi/4])  # Initial pan and tilt

# Simulate target movements
target1_trajectory = simulate_target_movement(target1_pos, delt, Nstep)
target2_trajectory = simulate_target_movement(target2_pos, delt, Nstep)

# Simulate camera tracking
projections1 = []
projections2 = []
camera_angles = []

for t in range(Nstep):
    proj1 = camera_projection(target1_trajectory[t], camera_state)
    proj2 = camera_projection(target2_trajectory[t], camera_state)
    projections1.append(proj1)
    projections2.append(proj2)
    # Assuming target1 is the primary target for simplicity
    camera_state = update_camera(camera_state, proj1, delt)
    camera_angles.append(camera_state)

projections1 = np.array(projections1)
projections2 = np.array(projections2)
camera_angles = np.array(camera_angles)

# Plotting results
plt.figure(figsize=(18, 6))
plt.subplot(131)
plt.plot(target1_trajectory[:, 0], target1_trajectory[:, 1], 'b-', label='Target 1')
plt.plot(target2_trajectory[:, 0], target2_trajectory[:, 1], 'r-', label='Target 2')
plt.title('Target Trajectories')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()

plt.subplot(132)
plt.plot(projections1[:, 0], projections1[:, 1], 'b-', label='Projection 1')
plt.plot(projections2[:, 0], projections2[:, 1], 'r-', label='Projection 2')
plt.title('Projections on Image Plane')
plt.xlabel('p_x')
plt.ylabel('p_y')
plt.legend()

plt.subplot(133)
plt.plot(camera_angles[:, 0], 'b-', label='Pan Angle')
plt.plot(camera_angles[:, 1], 'r-', label='Tilt Angle')
plt.title('Camera Angle Changes')
plt.xlabel('Time Step')
plt.ylabel('Angle (radians)')
plt.legend()

plt.tight_layout()
plt.show()
