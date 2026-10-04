import argparse
import json
import os
import time
from collections import deque

import cv2
import numpy as np

from config import DATASET_DIR, DANCE_CLASSES, FRAME_SIZE, SEQUENCE_LENGTH
from model import build_features, pad_or_trim
from pose_estimator import PoseEstimator
from stream_reader import pick_stream


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=None, help="video URL (auto-detects if omitted)")
    parser.add_argument("--label", default=None, choices=DANCE_CLASSES)
    args = parser.parse_args()

    if args.label is None:
        print("Classes:")
        for i, name in enumerate(DANCE_CLASSES):
            print(f"  {i + 1} = {name}")
        choice = input("Select class number (or type name): ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(DANCE_CLASSES):
            label = DANCE_CLASSES[int(choice) - 1]
        elif choice in DANCE_CLASSES:
            label = choice
        else:
            raise SystemExit("invalid class")
    else:
        label = args.label

    os.makedirs(DATASET_DIR, exist_ok=True)
    grabber = pick_stream(args.url)
    pose = PoseEstimator()
    buffer = deque(maxlen=SEQUENCE_LENGTH * 2)
    count = len([f for f in os.listdir(DATASET_DIR) if f.startswith(label + "_")])

    print(f"Recording label={label}. Keys: [s]=save last {SEQUENCE_LENGTH} frames, [c]=clear buffer, [q]=quit")
    last_seq = -1

    try:
        while True:
            frame, seq = grabber.latest()
            key = cv2.waitKey(10 if frame is None or seq == last_seq else 1) & 0xFF
            if key == ord("q"):
                break
            if frame is None or seq == last_seq:
                continue
            last_seq = seq

            frame = cv2.resize(frame, FRAME_SIZE)
            points = pose.process(frame)
            if points is not None:
                buffer.append(points)
                PoseEstimator.draw(frame, points)

            cv2.putText(frame, f"label: {label}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.putText(frame, f"buf: {len(buffer)}/{SEQUENCE_LENGTH}  fps: {grabber.fps:.1f}",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            cv2.imshow("collect", frame)

            if key == ord("c"):
                buffer.clear()
                print("buffer cleared")
            elif key == ord("s"):
                if len(buffer) < SEQUENCE_LENGTH:
                    print(f"need {SEQUENCE_LENGTH} frames, have {len(buffer)}")
                    continue
                seq_pts = np.stack(list(buffer)[-SEQUENCE_LENGTH:])
                feats = build_features(seq_pts)
                path = os.path.join(DATASET_DIR, f"{label}_{count:04d}.npy")
                np.save(path, feats)
                count += 1
                print(f"saved {path}")
    finally:
        grabber.stop()
        pose.close()
        cv2.destroyAllWindows()

    meta = {"classes": DANCE_CLASSES}
    with open(os.path.join(DATASET_DIR, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    main()
