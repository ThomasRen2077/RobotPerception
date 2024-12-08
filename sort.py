import numpy as np
from scipy.optimize import linear_sum_assignment
from kb import *
import matplotlib.pyplot as plt


class SORTTracker:
    """
    SORT tracker for multi-object tracking.
    """
    def __init__(self):
        self.trackers = []
        self.next_id = 0

    def update(self, detections):
        for tracker in self.trackers:
            tracker.predict()

        matches, unmatched_dets, unmatched_trks = self._match_detections_to_trackers(detections)

        for trk_idx, det_idx in matches:
            self.trackers[trk_idx].update(detections[det_idx])

        for det_idx in unmatched_dets:
            self.trackers.append(KalmanBoxTracker(detections[det_idx], self.next_id))
            self.next_id += 1

        self.trackers = [trk for trk in self.trackers if trk.time_since_update < 3]
        return [(trk.id, trk.get_state()) for trk in self.trackers]

    def _match_detections_to_trackers(self, detections):
        if not self.trackers:
            return [], list(range(len(detections))), []

        iou_matrix = np.zeros((len(detections), len(self.trackers)), dtype=np.float32)
        for d, det in enumerate(detections):
            for t, trk in enumerate(self.trackers):
                iou_matrix[d, t] = self._iou(det, trk.get_state())

        row_inds, col_inds = linear_sum_assignment(-iou_matrix)
        matches = [(col, row) for row, col in zip(row_inds, col_inds) if iou_matrix[row, col] >= 0.3]
        unmatched_dets = list(set(range(len(detections))) - {m[1] for m in matches})
        unmatched_trks = list(set(range(len(self.trackers))) - {m[0] for m in matches})
        return matches, unmatched_dets, unmatched_trks

    @staticmethod
    def _iou(bbox1, bbox2):
        xA = max(bbox1[0], bbox2[0])
        yA = max(bbox1[1], bbox2[1])
        xB = min(bbox1[2], bbox2[2])
        yB = min(bbox1[3], bbox2[3])
        interArea = max(0, xB - xA) * max(0, yB - yA)
        box1Area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        box2Area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        return interArea / float(box1Area + box2Area - interArea)


# # Initialize tracker
# tracker = SORTTracker()

# # Simulate bounding box detections over frames
# frames = [
#     [[100, 100, 150, 150], [200, 200, 250, 250]],  # Frame 1
#     [[105, 105, 155, 155], [205, 205, 255, 255]],  # Frame 2
#     [[110, 110, 160, 160], [210, 210, 260, 260]],  # Frame 3
#     [[300, 300, 350, 350]]                         # Frame 4: New object
# ]

# for frame_idx, detections in enumerate(frames):
#     tracked_objects = tracker.update(detections)
#     print(f"Frame {frame_idx + 1}")
#     for obj_id, bbox in tracked_objects:
#         print(f"ID: {obj_id}, BBox: {bbox}")
