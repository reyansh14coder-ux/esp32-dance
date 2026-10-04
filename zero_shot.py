import base64
import json
import os
import re
import threading
import time
import urllib.request

import numpy as np
from PIL import Image, ImageDraw

from config import (
    CLASSIFY_INTERVAL,
    GRID_FRAMES,
    GRID_SIZE,
    MIN_MOVEMENT,
    MIN_VISIBLE,
    NIM_API_URL,
    NIM_MODEL,
    SEQUENCE_LENGTH,
    ZERO_SHOT_CLASSES,
)
from pose_estimator import SKELETON_EDGES

PROMPT = (
    "You are an expert in Indian classical and folk dance. The image shows "
    f"{GRID_FRAMES} stick-figure frames of one front-facing dancer, in temporal "
    "order reading left-to-right then top-to-bottom, sampled over about 2 seconds. "
    "The figure is centered on the hips and scaled consistently across frames. "
    "Judge the dance form from the posture vocabulary: foot positions (aramandi "
    "half-squat, turnout), arm/mudra placement, torso angle, spin/chakkars, "
    "shoulder isolations, bouncy folk energy, and movement tempo.\n"
    "Respond with ONLY a JSON object, no other text:\n"
    '{"dance": "<label>", "confidence": <0-1>, "reason": "<max 15 words>"}\n'
    f"Allowed labels: {', '.join(ZERO_SHOT_CLASSES)}. "
    "Use 'none' if the person stands still, walks, sits, or is not dancing."
)


def _normalize(points):
    pts = points.astype(np.float32).copy()
    mid_hip = (pts[11, :2] + pts[12, :2]) / 2.0
    shoulder_mid = (pts[5, :2] + pts[6, :2]) / 2.0
    torso = np.linalg.norm(pts[11, :2] - shoulder_mid)
    if torso < 1e-6:
        torso = np.linalg.norm(pts[11, :2] - pts[12, :2]) or 1e-6
    pts[:, 0] = (pts[:, 0] - mid_hip[0]) / torso
    pts[:, 1] = (pts[:, 1] - mid_hip[1]) / torso
    return pts


def render_grid(points_seq, frames=GRID_FRAMES, size=GRID_SIZE):
    seq = np.asarray(points_seq, dtype=np.float32)
    if len(seq) > frames:
        idx = np.linspace(0, len(seq) - 1, frames).astype(int)
        seq = seq[idx]
    rows = cols = int(np.ceil(np.sqrt(frames)))
    cell = size // cols
    img = Image.new("RGB", (size, size), "white")
    draw = ImageDraw.Draw(img)

    for i, points in enumerate(seq):
        ox = (i % cols) * cell
        oy = (i // cols) * cell
        if (i // cols) >= rows:
            break
        normed = _normalize(points)
        px = np.empty((len(normed), 2), dtype=np.int32)
        px[:, 0] = np.clip(normed[:, 0] * (cell * 0.30) + ox + cell // 2, ox + 4, ox + cell - 4)
        px[:, 1] = np.clip(normed[:, 1] * (cell * 0.30) + oy + cell // 2, oy + 4, oy + cell - 4)
        for a, b in SKELETON_EDGES:
            if points[a, 2] > 0.3 and points[b, 2] > 0.3:
                draw.line([tuple(px[a]), tuple(px[b])], fill="black", width=4)
        for j in range(len(px)):
            if points[j, 2] > 0.3:
                x, y = px[j]
                draw.ellipse([x - 5, y - 5, x + 5, y + 5], fill="black")
        if i < cols:
            draw.line([ox, oy, ox + cell - 1, oy], fill=(210, 210, 210), width=2)
        draw.line([ox, oy, ox, oy + cell - 1], fill=(210, 210, 210), width=2)

    buf = __import__("io").BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def ask_vision(png_bytes, api_url=NIM_API_URL, model=NIM_MODEL, api_key=None):
    api_key = api_key or os.environ.get("NVIDIA_API_KEY", "")
    if not api_key:
        raise RuntimeError("NVIDIA_API_KEY not set")
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/png;base64,"
                            + base64.b64encode(png_bytes).decode()
                        },
                    },
                ],
            }
        ],
        "temperature": 0.1,
        "max_tokens": 250,
        "stream": False,
    }
    req = urllib.request.Request(
        api_url,
        data=json.dumps(payload).encode(),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.load(resp)
    return data["choices"][0]["message"]["content"]


def parse_answer(text):
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        obj = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    dance = str(obj.get("dance", "")).lower().strip()
    if dance not in ZERO_SHOT_CLASSES:
        return None
    try:
        conf = float(obj.get("confidence", 0.0))
    except (TypeError, ValueError):
        conf = 0.0
    return dance, min(max(conf, 0.0), 1.0), str(obj.get("reason", ""))[:80]


def movement_score(seq):
    arr = np.asarray(seq, dtype=np.float32)
    if len(arr) < 2:
        return 0.0
    disp = np.linalg.norm(arr[1:, :, :2] - arr[:-1, :, :2], axis=2)
    vis = (arr[1:, :, 2] > 0.3) & (arr[:-1, :, 2] > 0.3)
    if not vis.any():
        return 0.0
    per_step = (disp * vis).sum(axis=1) / np.maximum(vis.sum(axis=1), 1)
    return float(per_step.sum())


class ZeroShotDancer:
    def __init__(self, interval=CLASSIFY_INTERVAL, model=NIM_MODEL):
        self.interval = interval
        self.model = model
        self._buffer = []
        self._lock = threading.Lock()
        self._result = ("none", 0.0, "", 0.0)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._worker, daemon=True)

    def start(self):
        self._thread.start()

    def submit(self, points):
        with self._lock:
            self._buffer.append(points)
            if len(self._buffer) > SEQUENCE_LENGTH * 2:
                del self._buffer[: len(self._buffer) - SEQUENCE_LENGTH * 2]

    def result(self):
        with self._lock:
            return self._result

    def stop(self):
        self._stop.set()

    def _snapshot(self):
        with self._lock:
            if not self._buffer:
                return None
            return np.stack(self._buffer[-SEQUENCE_LENGTH:])

    def _worker(self):
        while not self._stop.is_set():
            time.sleep(self.interval)
            seq = self._snapshot()
            if seq is None:
                continue
            visible = int((seq[-1, :, 2] > 0.3).sum())
            move = movement_score(seq)
            if visible < MIN_VISIBLE or move < MIN_MOVEMENT:
                self._set_result("none", 0.9, f"idle (vis={visible}, move={move:.3f})")
                continue
            try:
                png = render_grid(seq)
                text = ask_vision(png, model=self.model)
                parsed = parse_answer(text)
                if parsed is None:
                    self._set_result("unknown", 0.0, text.strip()[:80])
                    continue
                dance, conf, reason = parsed
                self._set_result(dance, conf, reason)
            except Exception as e:
                self._set_result("error", 0.0, str(e)[:80])

    def _set_result(self, dance, conf, reason):
        with self._lock:
            self._result = (dance, conf, reason, time.time())
