import json

import numpy as np
import torch
import torch.nn as nn

from config import (
    DANCE_CLASSES,
    DROPOUT,
    HIDDEN_SIZE,
    LABEL_MAP_PATH,
    MODEL_PATH,
    NUM_KEYPOINTS,
    NUM_LAYERS,
    SEQUENCE_LENGTH,
)


def _frame_features(points):
    normed = points.copy()
    pts = normed[:, :2]
    mid_hip = (pts[11] + pts[12]) / 2.0
    shoulder_mid = (pts[5] + pts[6]) / 2.0
    torso = np.linalg.norm(pts[11] - shoulder_mid)
    if torso < 1e-6:
        torso = np.linalg.norm(pts[11] - pts[12]) or 1e-6
    normed[:, 0] = (pts[:, 0] - mid_hip[0]) / torso
    normed[:, 1] = (pts[:, 1] - mid_hip[1]) / torso
    normed[normed[:, 2] < 0.3, :2] = 0.0

    angles = np.zeros((NUM_KEYPOINTS, 1), dtype=np.float32)
    pairs = [
        (5, 7, 9), (6, 8, 10), (7, 5, 11), (8, 6, 12),
        (11, 13, 15), (12, 14, 16), (5, 11, 13), (6, 12, 14),
        (13, 11, 12), (14, 12, 11), (5, 6, 12), (6, 5, 11),
    ]
    for a, b, c in pairs:
        v1 = normed[a, :2] - normed[b, :2]
        v2 = normed[c, :2] - normed[b, :2]
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 < 1e-6 or n2 < 1e-6:
            continue
        cos = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
        angles[c, 0] = np.arccos(cos) / np.pi

    velocities = np.zeros_like(normed[:, :2])
    velocities[1:] = normed[1:, :2] - normed[:-1, :2]

    features = np.concatenate(
        [normed, angles, velocities], axis=1
    ).astype(np.float32)
    return features


def build_features(points):
    if points.ndim == 2:
        return _frame_features(points).reshape(-1)
    frames = [_frame_features(p) for p in points]
    return np.stack(frames).reshape(len(frames), -1)


def pad_or_trim(seq, length=SEQUENCE_LENGTH):
    if len(seq) >= length:
        return seq[-length:]
    pad = np.zeros((length - len(seq), seq.shape[1]), dtype=np.float32)
    return np.concatenate([pad, seq], axis=0)


def feature_size():
    return NUM_KEYPOINTS * 6





class DanceLSTM(nn.Module):
    def __init__(self, input_size=None, num_classes=len(DANCE_CLASSES)):
        input_size = input_size or feature_size()
        super().__init__()
        self.lstm = nn.LSTM(
            input_size,
            HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            batch_first=True,
            dropout=DROPOUT if NUM_LAYERS > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN_SIZE, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1])


def load_model(device=None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open(LABEL_MAP_PATH) as f:
        label_map = json.load(f)
    model = DanceLSTM(num_classes=len(label_map)).to(device)
    state = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state["model"])
    model.eval()
    return model, label_map, device
