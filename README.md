# ESP32-CAM Indian Dance Form Recognizer

Real-time Indian dance form recognition from an ESP32-CAM video stream using
MediaPipe pose estimation (17 COCO keypoints) and zero-shot AI classification.

No training data required — the built-in **auto mode** renders your pose
sequence as stick figures and asks a vision model to identify the dance form.

## Recognized Forms

bharatanatyam, kathak, odissi, kuchipudi, kathakali, mohiniyattam, manipuri,
sattriya, bhangra, garba, dandiya, lavani, chhau, bollywood, none

Edit `ZERO_SHOT_CLASSES` in `config.py` to change this list.

## Requirements

- Windows / Linux / macOS with Python 3.11+
- ESP32-CAM board (AI-Thinker) on the same Wi-Fi network
- `NVIDIA_API_KEY` environment variable (free key from https://build.nvidia.com)

```bash
pip install mediapipe opencv-python numpy pillow torch
```

## 1. Flash the ESP32-CAM

Use the `CameraWebServer` example from Arduino IDE (File → Examples →
ESP32 → Camera → CameraWebServer), set your Wi-Fi credentials, flash, then
open the serial monitor (115200 baud) to get the camera's IP.

The camera serves:
- `http://<IP>/` — control page (port 80)
- `http://<IP>:81/stream` — high-FPS MJPEG stream
- `http://<IP>/capture` — single JPEG snapshot (fallback, ~1 FPS)

## 2. Run (zero-shot, no training)

```bash
cd esp32-dance
python app.py
```

The app auto-probes both `:81/stream` and `/capture` and picks whichever
works. Force a source with:

```bash
python app.py --url http://192.168.1.50:81/stream
python app.py --url http://192.168.1.50/capture
python app.py --camera          # use webcam instead
```

On screen you see the skeleton overlay, the predicted dance form with
confidence, and the model's short reason. Recognition runs every ~2 seconds
in a background thread; idle scenes are gated to `none` locally without
calling the API.

Keys: `q` quit, `r` reset prediction.

## 3. Optional: train your own model (higher accuracy / lower latency)

Zero-shot is convenient but rough on similar forms (e.g. kuchipudi vs
bharatanatyam). A small trained LSTM fixes that:

```bash
python collect.py            # pick a class, press 's' to save 60-frame clips
python train.py              # trains dance_lstm.pt
python app.py --mode lstm    # run with the trained model
```

Collect 15–30 clips per class. Include `none` clips (standing/walking) so
the model learns to reject non-dance scenes.

## Configuration (`config.py`)

| Setting | Meaning |
|---|---|
| `STREAM_URL` / `CAPTURE_URL` | Camera endpoints |
| `ZERO_SHOT_CLASSES` | Labels offered to the AI classifier |
| `NIM_MODEL` | Vision model ID (NVIDIA NIM) |
| `CLASSIFY_INTERVAL` | Seconds between AI classifications |
| `SEQUENCE_LENGTH` | Frames per clip (60 ≈ 2–5 s) |
| `MIN_MOVEMENT` | Motion energy below this = `none` (no API call) |
| `DANCE_CLASSES` | Labels used by the trained LSTM |

## How It Works

```
ESP32-CAM MJPEG → MediaPipe Pose (17 COCO keypoints)
                        │
          ┌─────────────┴──────────────┐
     auto mode (default)         lstm mode
   background thread:            sliding window
   normalize → render 16-frame    → features → LSTM
   stick-figure grid → vision     → majority vote
   LLM → {dance, confidence}
```

- **Pose features**: keypoints are centered on the hips and scaled by torso
  length, plus 12 joint angles and per-joint velocities — so recognition is
  independent of distance and camera position.
- **Frame grabber**: video is read on a background thread so the UI never
  freezes on network stalls; results are majority-voted over the last 3
  predictions to prevent label flicker.

## Troubleshooting

| Problem | Fix |
|---|---|
| `no video source reachable` | Check camera power (5V 2A — weak supplies cause reboots), confirm IP, same Wi-Fi |
| `:81/stream` hangs | Only one client can stream — close other tabs using the camera, or reboot the ESP32; the app falls back to `/capture` automatically |
| IP changed after reboot | Check router device list or serial monitor |
| `NVIDIA_API_KEY not set` | Create a free key at build.nvidia.com, then `set NVIDIA_API_KEY=nvapi-...` |
| Low FPS | Use `:81/stream` instead of `/capture` (≈1 FPS), close other CPU-heavy apps |
| Wrong dance form | Zero-shot is approximate — collect data and use `--mode lstm` for reliability |
