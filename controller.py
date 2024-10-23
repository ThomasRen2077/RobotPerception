import numpy as np
from desired_angle import desired_angle

def controller(z, s, delt, u_prev, lambda_):
    b1 = b2 = 100 * np.pi / 180  # Conversion from radian per voltage-second squared
    PandT = desired_angle(z, s, lambda_)
    
    delpsi = PandT[0] - s[0]
    delphi = PandT[1] - s[1]
    
    # Compute the derivative of control inputs
    u_now1 = (delpsi / delt - s[2]) / b1
    u_now2 = (delphi / delt - s[3]) / b2
    u_now = np.array([u_now1, u_now2])
    
    # Compute the derivative of u_now with respect to time
    dudt = (u_now - u_prev) / delt
    
    # Proportional and derivative gains
    kp1 = kp2 = 0.3
    kd1 = kd2 = 0.03
    
    # PD control for each axis
    u1 = kp1 * u_now[0] + kd1 * dudt[0]
    u2 = kp2 * u_now[1] + kd2 * dudt[1]
    
    # Apply saturation limits to control signals
    u1 = np.clip(u1, -1, 1)
    u2 = np.clip(u2, -1, 1)
    
    return np.array([u1, u2])


# Example usage:
# z = np.array([0.1, 0.2, 0.15, 0.25])  # Target projections on the image plane
# s = np.array([0, np.pi / 4, 0.01, 0.01])  # Current state [pan, tilt, pan_rate, tilt_rate]
# delt = 0.1  # Time step
# u_prev = np.array([0.05, 0.05])  # Previous control outputs
# lambda_ = 4  # Focal length or similar parameter
# u = controller(z, s, delt, u_prev, lambda_)
# print("Control Outputs:", u)
