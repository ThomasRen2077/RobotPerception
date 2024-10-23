import numpy as np

def desired_angle(z, s, lambda_):
    # Extract projection coordinates for the targets
    px, py = z[0], z[1]
    px2, py2 = z[2], z[3]
    
    # Extracting camera angles
    psi, phi = s[0], s[1]
    
    # Define rotation matrices
    R_phi = np.array([
        [1, 0, 0],
        [0, np.cos(phi), np.sin(phi)],
        [0, -np.sin(phi), np.cos(phi)]
    ])

    R_psi = np.array([
        [np.cos(psi), -np.sin(psi), 0],
        [np.sin(psi), np.cos(psi), 0],
        [0, 0, 1]
    ])
    
    # The combined rotation matrix from world to camera
    C = np.array([
        R_phi[2, :] @ R_psi[:, 0],
        R_phi[2, :] @ R_psi[:, 1],
        R_phi[2, :] @ R_psi[:, 2]
    ])

    # Back-project to world coordinates
    P1 = np.array([px, py, lambda_])
    k1 = -lambda_ / (C @ P1)
    xT_world1 = k1 * (R_psi.T @ R_phi.T @ P1 - np.array([0, 0, -lambda_]))

    P2 = np.array([px2, py2, lambda_])
    k2 = -lambda_ / (C @ P2)
    xT_world2 = k2 * (R_psi.T @ R_phi.T @ P2 - np.array([0, 0, -lambda_]))

    # Calculate angles
    pan1 = np.pi/2 + np.arctan2(xT_world1[1], xT_world1[0])
    tilt1 = np.pi/2 + np.arctan2(lambda_, np.sqrt(xT_world1[0]**2 + xT_world1[1]**2))

    pan2 = np.pi/2 + np.arctan2(xT_world2[1], xT_world2[0])
    tilt2 = np.pi/2 + np.arctan2(lambda_, np.sqrt(xT_world2[0]**2 + xT_world2[1]**2))

    avg_pan = (pan1 + pan2) / 2
    avg_tilt = (tilt1 + tilt2) / 2

    return np.array([avg_pan, avg_tilt])

# Example usage
# z = np.array([0.1, 0.2, 0.15, 0.25])  # Simulated target projections
# s = np.array([0, np.pi/4])  # Initial camera angles (pan, tilt)
# lambda_ = 4  # Focal length
# angles = desired_angle(z, s, lambda_)
# print("Calculated Angles:", angles)
