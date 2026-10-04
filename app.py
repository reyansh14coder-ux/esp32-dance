import argparse
import os
import time
from collections import Counter, deque

import cv2
import numpy as np
import torch

from config import FRAME_SIZE, SEQUENCE_LENGTH, STRIDE
from model import build_features, load_model, pad_or_trim
from pose_estimator import PoseEstimator
from stream_reader import pick_stream
from zero_shot import ZeroShotDancer


class Predictor:
    def __init__(self):
        self.model, self.label_map, self.device = load_model()
        self.id_to_name = {v: k for k, v in self.label_map.items()}
        self.window = deque(maxlen=SEQUENCE_LENGTH)
        self.history = deque(maxlen=20)
        self.counter = 0

    def update(self, points):
        self.window.append(build_features(points))
        self.counter += 1
        if self.counter % STRIDE != 0 or len(self.window) < SEQUENCE_LENGTH // 2:
            return None
        seq = pad_or_trim(np.stack(self.window), SEQUENCE_LENGTH)
        x = torch.from_numpy(seq).unsqueeze(0).to(self.device)
        with torch.no_grad():
            probs = torch.softmax(self.model(x), dim=1).cpu().numpy()[0]
        idx = int(probs.argmax())
        self.history.append(idx)
        top = Counter(self.history).most_common(1)[0][0]
        return self.id_to_name[top], float(probs[idx])

    def reset(self):
        self.history.clear()
        self.window.clear()


class LabelSmoother:
    def __init__(self, window=3):
        self.votes = deque(maxlen=window)
        self.last_ts = 0.0

    def update(self, dance, conf, ts):
        if ts != self.last_ts:
            self.votes.append(dance)
            self.last_ts = ts
        if not self.votes:
            return dance, conf
        top = Counter(self.votes).most_common(1)[0][0]
        return top, conf if top == dance else 0.6

    def reset(self):
        self.votes.clear()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=None, help="video URL (auto-detects if omitted)")
    parser.add_argument("--camera", action="store_true", help="use webcam instead of ESP32")
    parser.add_argument("--mode", choices=["auto", "lstm"], default="auto",
                        help="auto = zero-shot AI recognition (no training), lstm = trained model")
    args = parser.parse_args()

    classifier, predictor, smoother = None, None, None
    if args.mode == "auto":
        classifier = ZeroShotDancer()
        classifier.start()
        smoother = LabelSmoother()
        print("auto mode: zero-shot recognition (no training needed)")
    else:
        if not os.path.exists("label_map.json") or not os.path.exists("dance_lstm.pt"):
            raise SystemExit("no trained model found - run collect.py + train.py, or use --mode auto")
        predictor = Predictor()
        print("lstm mode: using trained model")

    grabber = pick_stream(args.url, args.camera)
    pose = PoseEstimator()
    last_seq = -1
    current_label, current_conf, current_reason = None, 0.0, ""
    err_shown = False

    try:
        while True:
            frame, seq = grabber.latest()
            if frame is None or seq == last_seq:
                key = cv2.waitKey(10) & 0xFF
                if key == ord("q"):
                    break
                if grabber.last_error and not err_shown:
                    print(f"[video] {grabber.last_error}")
                    err_shown = True
                continue
            last_seq = seq
            err_shown = False

            frame = cv2.resize(frame, FRAME_SIZE)
            points = pose.process(frame)
            if points is not None:
                PoseEstimator.draw(frame, points)

            if classifier is not None:
                if points is not None:
                    classifier.submit(points)
                dance, conf, reason, ts = classifier.result()
                current_label, current_conf = smoother.update(dance, conf, ts)
                current_reason = reason
            elif predictor is not None and points is not None:
                result = predictor.update(points)
                if result is not None:
                    current_label, current_conf = result
                    current_reason = ""

            if current_label is not None:
                color = (0, 255, 0) if current_label not in ("none", "unknown", "error") else (0, 0, 255)
                text = f"{current_label}  {current_conf * 100:.0f}%"
                cv2.putText(frame, text, (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 3)
                if current_reason:
                    cv2.putText(frame, current_reason, (10, 70),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

            cv2.putText(frame, f"fps: {grabber.fps:.1f}  mode: {args.mode}",
                        (10, FRAME_SIZE[1] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.imshow("dance recognizer", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("r"):
                if predictor is not None:
                    predictor.reset()
                if smoother is not None:
                    smoother.reset()
                current_label, current_conf, current_reason = None, 0.0, ""
    finally:
        if classifier is not None:
            classifier.stop()
        grabber.stop()
        pose.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
