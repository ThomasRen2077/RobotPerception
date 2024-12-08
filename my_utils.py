import numpy as np
from matplotlib import pyplot as plt

########################
### Helper functions ###
########################

# 2D rotation
def get_2d_R(u):
    return np.array([
        [np.cos(u), -np.sin(u), 0],
        [np.sin(u), np.cos(u), 0],
        [0, 0, 1]
    ])

# phi rotation matrix
def get_R_phi(phi):
    R_phi = np.array(
    [[1, 0, 0],
     [0, np.cos(phi), np.sin(phi)],
     [0, -np.sin(phi), np.cos(phi)]]
    )
    return R_phi.T

# psi rotation matrix
def get_R_psi(psi):
    R_psi = np.array(
    [[np.cos(psi), np.sin(psi), 0],
     [-np.sin(psi), np.cos(psi), 0],
     [0, 0 , 1]]
    )
    return R_psi.T

# derivative phi rotation matrix
def get_R_phi_derivative(phi, phi_dot):
    R_phi = np.array(
    [[0, 0, 0],
     [0, -np.sin(phi), np.cos(phi)],
     [0, -np.cos(phi), -np.sin(phi)]]
    )
    return R_phi.T * phi_dot

# derivative psi rotation matrix, to be multiplied by psi_dot
def get_R_psi_derivative(psi, psi_dot):
    R_psi = np.array(
    [[-np.sin(psi), np.cos(psi), 0],
     [-np.cos(psi), -np.sin(psi), 0],
     [0, 0 , 0]]
    )
    return R_psi.T * psi_dot

# generator for intertial to virtual coordinate transformer
def intertial_to_virtual_coord_gen(psi, phi, f, x_c):
    # phi is tilt, psi is pan
    R_phi = get_R_phi(phi)
    R_psi = get_R_psi(psi)
    def i_to_v_transformer(x):
        q = (R_phi.T @ (R_psi.T @ (x-x_c).reshape(-1,1))).flatten()
        return f*np.array([q[0]/q[2], q[1]/q[2]])
    return i_to_v_transformer

# generator for virtual to inertial coordinate transformer
def virtual_to_inertial_coord(psi, phi, f, x_c):
    R_phi = get_R_phi(phi)
    R_psi = get_R_psi(psi)
    def v_to_i_transformer(p):
        p_aug = (
            R_psi @ R_phi @ (np.array([p[0]/f, p[1]/f, 1])).reshape(-1, 1)
        ).flatten()
        q_z = -x_c[2]/p_aug[2]
        return np.array([p_aug[0]*q_z+x_c[0], p_aug[1]*q_z+x_c[1], 0])
    return v_to_i_transformer

# generator for virtual to inertial coordinate transformer
def virtual_to_inertial_velocity_gen(psi, phi, psi_dot, phi_dot, f, x_c):
    R_phi = get_R_phi(phi)
    R_psi = get_R_psi(psi)
    R_phi_dot = get_R_phi_derivative(phi, phi_dot)
    R_psi_dot = get_R_psi_derivative(psi, psi_dot)
    v_to_i_xf = virtual_to_inertial_coord(psi, phi, f, x_c)
    def v_to_i_vel_transformer(p, p_dot):
        x = v_to_i_xf(p)
        q = (R_phi.T @ (R_psi.T @ (x-x_c).reshape(-1,1))).flatten()
        R_1 = R_psi @ R_phi
        R_2 = R_psi_dot @ R_phi + R_psi @ R_phi_dot
        delta_1 = np.array([p_dot[0]*q[2]/f, p_dot[1]*q[2]/f, 0])
        delta_2 = np.array([q[0]/q[2], q[1]/q[2], 1])
        q_z_dot = -(R_1 @ delta_1.reshape(-1, 1) + R_2 @ q.reshape(-1,1))[2,0]/(R_1 @ delta_2.reshape(-1, 1))[2, 0]
        q_dot = delta_1 + q_z_dot * delta_2
        x_dot = (R_1 @ q_dot.reshape(-1, 1) + R_2 @ q.reshape(-1, 1)).flatten()
        return x_dot
    return v_to_i_vel_transformer

def inertial_velocity_to_psi_phi_dot_gen(psi, phi, f, x_c):
    R_phi = get_R_phi(phi)
    R_psi = get_R_psi(psi)
    R_phi_dot = get_R_phi_derivative(phi, 1)
    R_psi_dot = get_R_psi_derivative(psi, 1)
    x = virtual_to_inertial_coord(psi, phi, f, x_c)([0, 0])
    q_z = (R_phi.T @ (R_psi.T @ (x-x_c).reshape(-1,1))).flatten()[2]
    def inertial_velocity_to_psi_phi_dot_xf(i_vel):
        R_0 = R_psi @ R_phi
        R_1 = R_psi_dot @ R_phi
        R_2 = R_psi @ R_phi_dot
        c_mat = np.zeros((2, 2))
        c_mat[:, 0] = R_1[:, -1][:2] - R_0[:, -1][:2]*R_1[-1, -1]/R_0[-1,-1]
        c_mat[:, 1] = R_2[:, -1][:2] - R_0[:, -1][:2]*R_2[-1, -1]/R_0[-1,-1]
        psi_dot, phi_dot = (np.linalg.inv(c_mat) @ (np.array(i_vel).reshape(-1, 1)/q_z)).flatten() 
        return psi_dot, phi_dot
    return inertial_velocity_to_psi_phi_dot_xf

########################
### Camera functions ###
########################

# Plot camera
def plot_camera(ax, x_c):
    # Plot camera
    u = np.linspace(0, 2*np.pi, 20)
    v = np.linspace(0, np.pi, 10)
    uc, vc=np.meshgrid(u, v)
    x = x_c[0]+0.5*np.cos(uc)*np.sin(vc)
    y = x_c[1]+0.5*np.sin(uc)*np.sin(vc)
    z = x_c[2]+0.5*np.cos(vc)
    ax.plot_surface(x, y, z, color="darkgray", shade=True)
    ax.text(x_c[0], x_c[1], x_c[2]+2, "PT Camera", fontsize=10)

class CameraFOV:
    
    def __init__(self, ax, psi, phi, psi_dot, phi_dot, f, sensor_w, sensor_h, x_c):
        self.psi = psi
        self.phi = phi
        self.psi_dot = psi_dot
        self.phi_dot = phi_dot
        self.f = f
        self.x_c = x_c
        self.sensor_w = sensor_w
        self.sensor_h = sensor_h
        self.ax = ax
        self.fov_corners, self.fov_collection, self.fov_lines = self._plot_FOV()
        
    def _plot_FOV(self):
        fov_data, fov_verts = self.get_FOV_measurements()
        fov_corners = self.ax.scatter3D(
            fov_data[:, 0], fov_data[:, 1], fov_data[:, 2], depthshade=False, s=1, color="C2", zorder=1)
        fov_collection = Poly3DCollection(fov_verts, facecolors="C2", linewidths=1, edgecolors="C2", alpha=.1, zorder=1)
        self.ax.add_collection3d(fov_collection)
        fov_lines = []
        for i in range(4):
            fov_line, = self.ax.plot3D(
                [self.x_c[0], fov_data[i][0]], 
                [self.x_c[1], fov_data[i][1]], 
                [self.x_c[2], fov_data[i][2]], color="C2", alpha=0.5, lw=1)
            fov_lines.append(fov_line)
        return  fov_corners, fov_collection, fov_lines
        
    def update_FOV(self, psi, phi):
        self.phi = phi
        self.psi = psi
        fov_data, fov_verts = self.get_FOV_measurements()
        self.fov_corners._offsets3d = (fov_data[:, 0], fov_data[:, 1], fov_data[:, 2])
        self.fov_collection.set_verts(fov_verts)
        for i, fov_line in enumerate(self.fov_lines):
            update_data = np.vstack(([self.x_c], fov_data[[i], :])).T
            fov_line.set_data(
                update_data[:2, :]
                )
            fov_line.set_3d_properties(update_data[2, :])
        return self
        
    def get_FOV_measurements(self):
        v_to_i_tx = virtual_to_inertial_coord(self.psi, self.phi, self.f, self.x_c)
        fov_data = np.array([
            v_to_i_tx(np.array([self.sensor_w/2, self.sensor_h/2])),
            v_to_i_tx(np.array([self.sensor_w/2, -self.sensor_h/2])),
            v_to_i_tx(np.array([-self.sensor_w/2, self.sensor_h/2])),
            v_to_i_tx(np.array([-self.sensor_w/2, -self.sensor_h/2]))
        ])
        fov_verts = [ 
            [fov_data[0],fov_data[1],fov_data[3], fov_data[2]]
        ]
        return fov_data, fov_verts
