import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
from matplotlib import pyplot as plt


### Camera functions ###

# Plot camera
def plot_camera(ax, x_c):
    u = np.linspace(0, 2*np.pi, 20)
    v = np.linspace(0, np.pi, 10)
    uc, vc=np.meshgrid(u, v)
    x = x_c[0]+0.5*np.cos(uc)*np.sin(vc)
    y = x_c[1]+0.5*np.sin(uc)*np.sin(vc)
    z = x_c[2]+0.5*np.cos(vc)
    ax.plot_surface(x, y, z, color="red", shade=True)
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
    
    def get_FOV_direction(self):
        sensor_center = np.array([0, 0])  
        v_to_i_tx = virtual_to_inertial_coord(self.psi, self.phi, self.f, self.x_c)
        center_world = v_to_i_tx(sensor_center)
        direction = center_world - np.array(self.x_c)
        direction_normalized = direction / np.linalg.norm(direction)
        return direction_normalized



### Vehicle functions ###

class LinearVehicle:
    
    def __init__(self, ax, ax2, v_l=3, v_w=3, x_offset=0, y_offset=0, vel_dir=[1, 1], color="C0", is_visible=True):
        self.ax = ax
        self.ax2 = ax2
        self.v_l = v_l
        self.v_w = v_w
        self.x_offset = x_offset
        self.y_offset = y_offset
        self.vel_dir = vel_dir
        self.color = color
        # Initialize vehicles
        #self.v_vel = np.random.uniform(0.5, 1.5) * np.array(vel_dir)
        self.v_vel = np.array(vel_dir)
        self.v_data_0, self.v_corners, self.v_collection = self._plot_vehicle()
        self.vv_corners, self.vv_collection = None, None
        self.is_visible = is_visible
        if not self.is_visible:
            self.set_visible(False)
            
    def _plot_vehicle(self):
        # Inertial dataframe
        v_data_init = np.array([[self.v_w, -self.v_l/2, 0], 
                   [self.v_w, self.v_l/2, 0], 
                   [0, -self.v_l/2, 0], 
                   [0, +self.v_l/2, 0]])
        R_mat = get_2d_R(np.arctan(self.v_vel[1]/self.v_vel[0]))
        v_data = (R_mat @ v_data_init.T).T
        v_data[:, 0] += self.x_offset
        v_data[:, 1] += self.y_offset
        v_corners = self.ax.scatter3D(
            v_data[:, 0], v_data[:, 1], v_data[:, 2], depthshade=False, s=1, color=self.color, zorder=10)
        v_verts = [ 
            [v_data[0],v_data[1],v_data[3], v_data[2]]
        ]
        v_collection = Poly3DCollection(v_verts, 
         facecolors=self.color, linewidths=1, edgecolors=self.color, alpha=.5, zorder=10)
        self.ax.add_collection3d(v_collection)
        return v_data, v_corners, v_collection
    
    def update_vehicle(self, t):
        v_data = self.get_vehicle_data(t)
        v_verts = [ 
                [v_data[0],v_data[1],v_data[3], v_data[2]]
            ]
        self.v_corners._offsets3d = (v_data[:, 0], v_data[:, 1], v_data[:, 2])
        self.v_collection.set_verts(v_verts)
        return self
    
    def update_vehicle_virtual(self, camera_fov, t):
        vv_data = self.get_vehicle_data_virtual(camera_fov, t)
        vv_verts = [ 
                [vv_data[0],vv_data[1],vv_data[3], vv_data[2]]
            ]
        if self.vv_corners is None:
            self.vv_corners = self.ax2.scatter(vv_data[:, 0], vv_data[:, 1], color=self.color, s=1)
            self.vv_collection = collections.PolyCollection(
                vv_verts, 
                facecolors=self.color, linewidths=1, edgecolors=self.color, alpha=.5)
            self.ax2.add_collection(self.vv_collection, autolim=True)
        else:
            self.vv_corners.set_offsets(vv_data)
            self.vv_collection.set_verts(vv_verts)
        return self
    
    def get_vehicle_data(self, t):
        v_data = np.zeros((4, 3))
        v_data[:, 0] = self.v_data_0[:, 0] + self.v_vel[0]*t
        v_data[:, 1] = self.v_data_0[:, 1] + self.v_vel[1]*t
        return v_data
    
    def get_vehicle_data_virtual(self, camera_fov, t):
        v_data = self.get_vehicle_data(t)
        psi, phi, f, x_c = camera_fov.psi, camera_fov.phi, camera_fov.f, camera_fov.x_c
        i_to_v_tx = intertial_to_virtual_coord_gen(psi, phi, f, x_c)
        vv_data = np.zeros((4, 2))
        for i, vec in enumerate(v_data):
            vv_data[i] = i_to_v_tx(v_data[i])
        return vv_data

    def set_visible(self, visibility=False):
        self.v_collection.set_visible(visibility)
        self.v_corners.set_visible(visibility)
        if self.vv_corners is not None and self.vv_collection is not None:
            self.vv_corners.set_visible(visibility)
            self.vv_collection.set_visible(visibility)
        self.is_visible = visibility
        return self

### WildAnimal functions ###

class WildAnimal:
    def __init__(self, ax, x_init, y_init, z_init=0, color="C1"):
        self.ax = ax
        self.position = np.array([x_init, y_init, z_init], dtype=np.float64)
        self.color = color
        self.speed = np.random.uniform(0.05, 0.2)  # Random initial speed
        self.direction = np.random.uniform(0, 2 * np.pi)  # Random initial direction
        self.pause_probability = 0.1  # Probability of pausing
        self.corners, self.collection = self._plot_animal()

    def _plot_animal(self):
        animal_data = np.array([[self.position[0] - 0.5, self.position[1] - 0.5, self.position[2]],
                                [self.position[0] + 0.5, self.position[1] - 0.5, self.position[2]],
                                [self.position[0] + 0.5, self.position[1] + 0.5, self.position[2]],
                                [self.position[0] - 0.5, self.position[1] + 0.5, self.position[2]]])
        corners = self.ax.scatter3D(animal_data[:, 0], animal_data[:, 1], animal_data[:, 2], color=self.color, s=1)
        verts = [[animal_data[0], animal_data[1], animal_data[2], animal_data[3]]]
        collection = Poly3DCollection(verts, facecolors=self.color, linewidths=1, edgecolors=self.color, alpha=0.5)
        self.ax.add_collection3d(collection)
        return corners, collection

    def update(self):
        # Randomly decide whether to pause
        if np.random.rand() < self.pause_probability:
            return

        # Update direction with a random small perturbation
        self.direction += np.random.uniform(-np.pi / 8, np.pi / 8)

        # Update position
        dx = self.speed * np.cos(self.direction)
        dy = self.speed * np.sin(self.direction)
        new_position = self.position + np.array([dx, dy, 0], dtype=np.float64)

        # Enforce boundary constraints
        if 0 <= new_position[0] <= 40 and 0 <= new_position[1] <= 40:
            self.position = new_position
        else:
            # Reflect direction upon hitting boundary
            self.direction += np.pi  # Reverse direction

        # Update visualization
        animal_data = np.array([[self.position[0] - 0.5, self.position[1] - 0.5, self.position[2]],
                                [self.position[0] + 0.5, self.position[1] - 0.5, self.position[2]],
                                [self.position[0] + 0.5, self.position[1] + 0.5, self.position[2]],
                                [self.position[0] - 0.5, self.position[1] + 0.5, self.position[2]]])
        verts = [[animal_data[0], animal_data[1], animal_data[2], animal_data[3]]]
        self.corners._offsets3d = (animal_data[:, 0], animal_data[:, 1], animal_data[:, 2])
        self.collection.set_verts(verts)

### Helper functions ###

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
