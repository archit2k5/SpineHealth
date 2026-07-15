# SpineHealth — Privacy-First Edge AI Posture & Occupancy Monitor

On-device posture and desk-occupancy detector. Runs entirely on a Raspberry Pi 5 — no video or keypoint data leaves the device.

## Hardware
- Raspberry Pi 5 (8GB)
- Camera: Waveshare IMX219 (via Picamera2 / libcamera)
- 2x LEDs — Red: GPIO 17 (physical pin 11), Green: GPIO 27 (physical pin 13), with 220Ω current-limiting resistors, shared GND

## Software / Model
- YOLOv8n-pose, exported to NCNN for ARM CPU inference
- Picamera2 for camera capture, OpenCV for debug overlay/display
- gpiozero for GPIO, SQLite3 for local event logging, NumPy for centroid math

## Setup
```bash
sudo apt update && sudo apt install -y python3-pip python3-opencv
pip3 install ultralytics gpiozero picamera2 numpy
```
Export and place the NCNN model:
```bash
python3 -c "from ultralytics import YOLO; YOLO('yolov8n-pose.pt').export(format='ncnn')"
# copy resulting yolov8n-pose_ncnn_model/ into the project directory
```

## Run
```bash
python3 main.py
```
- `DEBUG_DISPLAY = True`: shows a local annotated preview window (ratio + status overlay) for demo purposes; press `q` to quit. This window is a live visualization only — nothing is saved or transmitted.
- `DEBUG_DISPLAY = False`: headless mode, processes every 5th frame for lower CPU load.

## How it works
- YOLOv8n-pose (NCNN) extracts 17 keypoints per detected person.
- Computes Normalized Posture Ratio (NPR) = vertical head-to-shoulder distance / shoulder width, using confidence-weighted centroids.
- 3-tier fallback for shoulder width if occluded: live shoulders → EMA of recent shoulder width → ear-width estimate (×2.1).
- NPR < 0.6 sustained for 5s → logs `SLOUCH_DETECTED`, red LED on.
- Absence > 60s → `DESK_DEPARTURE`; presence → `DESK_ARRIVAL`. All logged to local SQLite (`workspace_metrics.db`) with timestamp, event type, and NPR value only.

## Configuration
| Parameter | Value |
|---|---|
| SLOUCH_RATIO_THRESHOLD | 0.6 |
| SLOUCH_TIME_THRESHOLD | 5s |
| BREAK_TIME_THRESHOLD | 60s |
| CONF_THRESHOLD | 0.35 |
| FRAME_SKIP | 1 (debug) / 5 (headless) |

## Sample output
```
Logged: DESK_ARRIVAL | Value: 0.00
Logged: SLOUCH_DETECTED | Value: 0.52
Logged: DESK_DEPARTURE | Value: 0.00
```

## Known limitations
- Single-person tracking only.
- NPR catches forward head/neck collapse but can miss whole-torso lean when the back segment itself stays locally straight (see Evaluation doc).
- Thresholds are heuristic, tuned by manual calibration, not clinically validated.

## Attribution
- Ultralytics YOLOv8n-pose (AGPL-3.0)
- NCNN (Tencent)
- Picamera2 (Raspberry Pi Foundation)
- gpiozero

## Demo Video
[![Watch the video](https://images.placeholders.dev/?width=640&height=360&text=Click%20to%20Play%20Video&bgColor=%23282c34&textColor=%2361dafb)](https://drive.google.com/file/d/1TP_Pb0A-607R76k7WZC460XgktkaaU--/view)

