import numpy as np

def kinematic_cam(s_k, u_k, delt):
    psiDot_m = 100 * np.pi / 180  # Maximum angular velocity [rad/s]
    phiDot_m = 100 * np.pi / 180  # Maximum angular velocity [rad/s]
    b1 = b2 = 100 * np.pi / 180  # Control input to angular velocity conversion factor

    # State transition matrix
    A = np.array([[1, 0, delt, 0],
                  [0, 1, 0, delt],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]])
    
    # Control input matrix
    B = np.array([[0, 0],
                  [0, 0],
                  [b1, 0],
                  [0, b2]])

    # Compute the next raw state
    s_next_raw = A @ s_k + B @ u_k

    # Constraints for each state component
    b1_vec = np.array([0, np.pi / 2, -psiDot_m, -phiDot_m])
    b2_vec = np.array([2 * np.pi, np.pi, psiDot_m, phiDot_m])

    # Apply saturation to ensure state limits are respected
    s_next = np.minimum(b2_vec, np.maximum(b1_vec, s_next_raw))

    return s_next

# Example usage
# s_k = np.array([0, np.pi/4, 0.1, 0.1])  # Current state [pan angle, tilt angle, pan rate, tilt rate]
# u_k = np.array([0.05, 0.05])  # Control inputs
# delt = 0.1  # Time step
# s_next = kinematic_cam(s_k, u_k, delt)
# print("Next State:", s_next)
