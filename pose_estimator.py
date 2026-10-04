import cv2
import mediapipe as mp
import numpy as np

from config import NUM_KEYPOINTS

BLAZETOPOSE_TO_COCO = [
    0, 2, 5, 7, 8,
    11, 12, 13, 14, 15, 16,
    23, 24, 25, 26, 27, 28,
]

SKELETON_EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]

JOINT_ANGLES = [
    (5, 7, 9), (6, 8, 10),
    (7, 5, 11), (8, 6, 12),
    (11, 13, 15), (12, 14, 16),
    (5, 11, 13), (6, 12, 14),
    (11, 13, 15), (12, 14, 16),
    (13, 11, 12), (14, 12, 11),
]


class PoseEstimator:
    def __init__(self, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self._pose = mp.solutions.pose.Pose(
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def process(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        result = self._pose.process(rgb)
        if not result.pose_landmarks:
            return None
        return self._extract(result.pose_landmarks)

    def _extract(self, landmarks):
        points = np.zeros((NUM_KEYPOINTS, 3), dtype=np.float32)
        for out_idx, blaze_idx in enumerate(BLAZETOPOSE_TO_COCO):
            lm = landmarks.landmark[blaze_idx]
            points[out_idx] = (lm.x, lm.y, lm.visibility)
        return points

    @staticmethod
    def normalize(points):
        pts = points.copy()
        visible = pts[:, 2] > 0.3
        hip_l, hip_r = pts[11], pts[12]
        mid_hip = (hip_l[:2] + hip_r[:2]) / 2.0
        shoulder_mid = (pts[5][:2] + pts[6][:2]) / 2.0
        torso = np.linalg.norm(hip_l[:2] - shoulder_mid)
        if torso < 1e-6:
            torso = np.linalg.norm(hip_l[:2] - hip_r[:2]) or 1e-6
        pts[:, 0] = (pts[:, 0] - mid_hip[0]) / torso
        pts[:, 1] = (pts[:, 1] - mid_hip[1]) / torso
        pts[~visible, :2] = 0.0
        return pts

    @staticmethod
    def joint_angles(points):
        angles = np.zeros(len(JOINT_ANGLES), dtype=np.float32)
        for i, (a, b, c) in enumerate(JOINT_ANGLES):
            v1 = points[a, :2] - points[b, :2]
            v2 = points[c, :2] - points[b, :2]
            n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
            if n1 < 1e-6 or n2 < 1e-6:
                angles[i] = 0.0
                continue
            cos = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
            angles[i] = np.degrees(np.arccos(cos))
        return angles

    @staticmethod
    def draw(frame, points, color=(0, 255, 0)):
        h, w = frame.shape[:2]
        px = (points[:, :2] * [w, h]).astype(int)
        for i, j in SKELETON_EDGES:
            if points[i, 2] > 0.3 and points[j, 2] > 0.3:
                cv2.line(frame, tuple(px[i]), tuple(px[j]), color, 2)
        for idx, p in enumerate(px):
            if points[idx, 2] > 0.3:
                cv2.circle(frame, tuple(p), 4, (0, 165, 255), -1)
        return frame

    def close(self):
        self._pose.close()
