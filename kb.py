from filterpy.kalman import KalmanFilter
import numpy as np

class KalmanBoxTracker:
    """
    Represents a tracked object using a Kalman filter.
    """
    def __init__(self, bbox, tracker_id):
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.eye(7)  # State transition matrix
        self.kf.H = np.eye(4, 7)  # Measurement function
        self.kf.R[2:, 2:] *= 10.0  # Measurement noise
        self.kf.P[4:, 4:] *= 1000.0  # Initial uncertainty
        self.kf.P *= 10.0
        self.kf.Q[-1, -1] *= 0.01  # Process noise
        self.kf.Q[4:, 4:] *= 0.01

        cx, cy, w, h = self._convert_bbox_to_center(bbox)
        self.kf.x[:4] = np.array([cx, cy, w, h]).reshape(-1, 1)
        self.time_since_update = 0
        self.id = tracker_id
        self.hits = 0
        self.age = 0

    def predict(self):
        self.kf.predict()
        self.age += 1
        self.time_since_update += 1

    def update(self, bbox):
        cx, cy, w, h = self._convert_bbox_to_center(bbox)
        self.kf.update(np.array([cx, cy, w, h]))
        self.time_since_update = 0
        self.hits += 1

    def get_state(self):
        return self._convert_center_to_bbox(self.kf.x[:4].flatten())

    @staticmethod
    def _convert_bbox_to_center(bbox):
        x_min, y_min, x_max, y_max = bbox
        cx = (x_min + x_max) / 2
        cy = (y_min + y_max) / 2
        w = x_max - x_min
        h = y_max - y_min
        return cx, cy, w, h

    @staticmethod
    def _convert_center_to_bbox(center):
        cx, cy, w, h = center
        x_min = cx - w / 2
        y_min = cy - h / 2
        x_max = cx + w / 2
        y_max = cy + h / 2
        return [x_min, y_min, x_max, y_max]
