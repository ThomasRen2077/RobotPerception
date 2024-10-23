import numpy as np

def measurement_cam(s, xT, xTdot, xT2, xTdot2, lambda_):
    # Camera position and states
    xC = np.array([0, 0, 4])
    psi, phi, psiDot, phiDot = s

    # Rotation matrices for pan and tilt
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

    R = R_psi @ R_phi
    R_inv = R.T

    # Transform targets to camera coordinate system
    qT = R_inv @ (np.hstack([xT, [0]]) - xC)
    qT2 = R_inv @ (np.hstack([xT2, [0]]) - xC)

    # Project onto image plane
    pT = lambda_ * (qT[:2] / qT[2])
    pT2 = lambda_ * (qT2[:2] / qT2[2])

    # Image Jacobian matrices
    H = calculate_jacobian(pT, qT, lambda_)
    H2 = calculate_jacobian(pT2, qT2, lambda_)

    # Calculate velocities on image plane
    R_6 = np.block([
        [R_phi.T @ R_psi.T, np.zeros((3, 3))],  # Upper blocks
        [np.zeros((3, 3)), -R_phi.T]            # Lower blocks
    ])

    pTdot = H @ R_6 @ np.hstack([xTdot, 0, phiDot, 0, psiDot])
    pTdot2 = H2 @ R_6 @ np.hstack([xTdot2, 0, phiDot, 0, psiDot])

    # Construct output
    z = np.hstack([pT, pTdot])
    z2 = np.hstack([pT2, pTdot2])

    return z, z2

def calculate_jacobian(pT, qT, lambda_):
    px, py = pT
    qz = qT[2]
    return np.array([
        [-lambda_ / qz, 0, px / qz, px * py / lambda_, -(lambda_**2 + px**2) / lambda_, py],
        [0, -lambda_ / qz, py / qz, (lambda_**2 + px**2) / lambda_, -px * py / lambda_, -px]
    ])

def test_measurement_cam():
    # Define camera settings
    lambda_ = 4  
    s = np.array([0, np.pi/4, 0, 0])  

    # Define target positions
    xT = np.array([1, 0.5])  
    xT2 = np.array([-1, 1.5]) 

    # Define target velocities
    xTdot = np.array([0.1, 0.2])  
    xTdot2 = np.array([0.15, 0.25])  

    # Call the measurement function
    z, z2 = measurement_cam(s, xT, xTdot, xT2, xTdot2, lambda_)

    # Print the outputs
    print("Projection and velocity for the first target:", z)
    print("Projection and velocity for the second target:", z2)

# test_measurement_cam()
