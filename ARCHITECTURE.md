# Architecture

## Pipeline
```
IMX219 Camera (Picamera2)
        |
YOLOv8n-pose (NCNN, ARM CPU) -> 17 keypoints (x, y, conf)
        |
compute_posture_index()
  - confidence-weighted head centroid (nose, ears)
  - confidence-weighted shoulder centroid
  - d_v = shoulder_y - head_y
  - d_h = shoulder width, 3-tier fallback:
      1. live L/R shoulder distance
      2. EMA of recent shoulder width (alpha=0.1), if shoulders low-conf
      3. ear-to-ear width x 2.1, cold start
  - NPR = d_v / d_h
        |
Threshold logic (NPR < 0.6 for 5s -> slouch)
        |            |
   Red/Green LED   SQLite log (workspace_metrics.db)
```

## Components
- **Capture**: Picamera2, 1280x720 native, resized to 640x480 for inference.
- **Inference**: YOLOv8n-pose exported to NCNN, runs on Pi 5 CPU (no GPU/NPU used).
- **Posture logic**: `compute_posture_index()` — stateless per-frame except for the shoulder-width EMA, held in a `tracker_state` dict across frames.
- **Occupancy logic**: presence/absence tracked via `last_seen_time`; departure logged only after 60s continuous absence to avoid false departures from brief missed detections.
- **Output**: GPIO LEDs (immediate feedback) + SQLite (persistent local log).

## Local vs cloud
Fully local. No network calls anywhere in the pipeline — capture, inference, logic, and storage all run on-device.

## Key design decisions
- **Scale-normalized ratio (NPR)** instead of raw pixel distance, so posture detection doesn't depend on distance from camera or person size.
- **3-tier fallback chain** for shoulder width so brief occlusion (e.g. arm raised) doesn't break detection.
- **Confidence gating (0.35 threshold)** zeroes out unreliable keypoints before they pollute centroid calculations.
- **Debounced logging**: slouch state only logs once per SLOUCH_TIME_THRESHOLD interval, not every frame, to avoid flooding the DB.
- **DEBUG_DISPLAY flag**: local preview window for demo/dev, separate code path from the headless deployment mode — does not affect what's stored.
