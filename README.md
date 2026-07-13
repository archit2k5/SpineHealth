# 🦴 SpineHealth — Privacy-First Edge AI Posture & Occupancy Monitor

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Platform](https://img.shields.io/badge/Platform-Raspberry%20Pi%205-C51A4A?style=for-the-badge&logo=raspberry-pi&logoColor=white)](https://www.raspberrypi.com/products/raspberry-pi-5/)
[![Model](https://img.shields.io/badge/AI%20Model-YOLOv8n--pose-00FFFF?style=for-the-badge&logo=pytorch&logoColor=black)](https://docs.ultralytics.com/models/yolov8/)
[![Inference](https://img.shields.io/badge/Inference-NCNN%20ARM-FF6F00?style=for-the-badge&logo=c%2B%2B&logoColor=white)](https://github.com/Tencent/ncnn)
[![Storage](https://img.shields.io/badge/Storage-SQLite3-003B57?style=for-the-badge&logo=sqlite&logoColor=white)](https://www.sqlite.org/)
[![Privacy](https://img.shields.io/badge/Privacy-100%25%20Offline-2ECC71?style=for-the-badge&logo=shield&logoColor=white)]()
[![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)
[![Stars](https://img.shields.io/github/stars/archit2k5/SpineHealth?style=for-the-badge&color=gold&logo=github)](https://github.com/archit2k5/SpineHealth/stargazers)

</div>

> **An offline, AI-powered desk monitor that tracks posture and occupancy in real-time — entirely on-device, with zero data ever leaving your hardware.**

<div align="center">

```
 Camera  →   YOLOv8n-pose (NCNN)  →   NPR Algorithm  →   LED Feedback  →   SQLite Log
     ↑                                                                                        ↓
  IMX219                                                                          workspace_metrics.db
  640×480                                                                        (on-device only)
```

</div>

---

## Table of Contents

1. [Project Overview](#-project-overview)
2. [Key Design Principles](#-key-design-principles)
3. [Hardware Stack](#-hardware-stack)
4. [Hardware Wiring](#-hardware-wiring)
5. [Software & AI Stack](#-software--ai-stack)
6. [System Architecture](#-system-architecture)
7. [The Mathematical Core — Normalized Posture Ratio (NPR)](#-the-mathematical-core--normalized-posture-ratio-npr)
8. [Code Walkthrough](#-code-walkthrough)
   - [1. Database Initialization](#1-database-initialization)
   - [2. Posture Math & Keypoint Processing](#2-posture-math--keypoint-processing)
   - [3. Edge Execution Pipeline](#3-edge-execution-pipeline)
9. [State Machines](#-state-machines)
10. [Configuration & Tuning](#-configuration--tuning)
11. [Privacy Guarantee](#-privacy-guarantee)
12. [Installation & Setup](#-installation--setup)
13. [Running the Monitor](#-running-the-monitor)
14. [Database Schema & Event Logging](#-database-schema--event-logging)
15. [Performance Optimizations](#-performance-optimizations)
16. [Calibration Guide](#-calibration-guide)
17. [Troubleshooting](#-troubleshooting)
18. [Project Structure](#-project-structure)

---

## Project Overview

**SpineHealth** is a fully offline, privacy-preserving edge AI system built for desk environments. Mounted above or beside your workstation, it uses a Raspberry Pi 5 and a camera to:

- **Detect occupancy** — log when a user arrives at or departs from the desk.
- **Monitor posture in real-time** — detect slouching events using a biomechanically-inspired algorithm.
- **Provide immediate physical feedback** — a **green LED** signals good posture; a **red LED** activates when slouching persists beyond a configurable time window.
- **Log all events locally** — every occupancy and posture event is persisted to an on-device SQLite database.

The critical architectural decision: **video frames are never written to disk, never transmitted over a network, and are immediately discarded from RAM after keypoint extraction.** The system only ever stores the mathematical output (a single floating-point ratio), not any visual data.

---

## Key Design Principles

| Principle | Implementation |
|---|---|
| **Privacy by Design** | No `cv2.imshow()`, no `cv2.imwrite()`. Frames are read → analyzed → discarded in a single loop iteration. |
| **Offline-First** | No cloud calls, no internet dependency. SQLite is a single file on the SD card. |
| **Edge-Optimized** | NCNN inference engine converts YOLOv8 to ARM-native format; frame skipping reduces CPU load to ~20%. |
| **Robust Measurement** | A 3-tier fallback chain ensures posture measurement remains valid even when shoulders are partially occluded. |
| **Safe Hardware Control** | `gpiozero` provides safe, high-level GPIO access with automatic cleanup on exit. |

---

## Hardware Stack

| Component | Details |
|---|---|
| **Compute Board** | Raspberry Pi 5 (4GB or 8GB RAM recommended) |
| **Camera** | Waveshare IMX219 Camera Module (connected via CSI ribbon cable) |
| **Visual Feedback** | 1× Red LED + 1× Green LED |
| **Passive Components** | 2× ~220Ω resistors (for LED current limiting) |
| **Prototyping** | Half-size breadboard + jumper wires |

### Why the Raspberry Pi 5?
The Pi 5 features a significantly upgraded ARM Cortex-A76 CPU (vs. the Cortex-A72 in Pi 4), which provides roughly 2–3× the single-core performance. Combined with the NCNN inference framework, this is sufficient to run YOLOv8n-pose at ~6 effective FPS with low thermal throttling.

### Why the IMX219?
The IMX219 is a Sony sensor natively supported by the Pi's `libcamera` stack. It is compact, lightweight, and runs efficiently at low resolutions (640×480), reducing the data throughput that OpenCV must process on each frame.

---

## Hardware Wiring

The GPIO pins use **BCM numbering** internally (as referenced by `gpiozero`), but below are the **Physical (Board)** pin numbers for ease of breadboard wiring.

```
Raspberry Pi 5 GPIO Header
─────────────────────────────────────────────────

[Physical Pin 11]  →  GPIO 17  →  220Ω Resistor  →  RED LED Anode  →  GND
[Physical Pin 13]  →  GPIO 27  →  220Ω Resistor  →  GREEN LED Anode →  GND
[Physical Pin 6 or 9]  →  GND  →  LED Cathodes (both share ground)

─────────────────────────────────────────────────
```

**Wiring Schematic (simplified):**
```
Pi GPIO 17 (Pin 11) ──[220Ω]──▶|── GND
                                  RED LED

Pi GPIO 27 (Pin 13) ──[220Ω]──▶|── GND
                                  GREEN LED
```

> **Important:** Always use a current-limiting resistor. The Pi's GPIO pins source ~3.3V; without a resistor, an LED will draw excessive current and may damage the GPIO pin permanently.

---

## Software & AI Stack

| Technology | Role | Why This Choice |
|---|---|---|
| **Python 3.11+** | Core language | Excellent ecosystem for AI, CV, and hardware I/O |
| **Ultralytics YOLOv8n-pose** | Pose estimation model | "Nano" variant is the smallest/fastest; 17-keypoint COCO body skeleton |
| **NCNN** | Inference engine | ARM-optimized C++ runtime; dramatically faster than ONNX/TFLite on Pi CPU |
| **OpenCV (`cv2`)** | Camera capture | Industry-standard, `libcamera` compatible via V4L2 backend on Pi |
| **gpiozero** | GPIO control | High-level, safe abstraction; handles pin cleanup automatically |
| **SQLite3** | Event storage | Serverless, zero-config, single-file database; perfect for edge logging |
| **NumPy** | Array math | Vectorized centroid and weight calculations |

### YOLOv8n-pose: The 17 COCO Keypoints
The model outputs 17 anatomical landmarks per detected person, each as `(x, y, confidence)`:

```
Index │ Keypoint       │ Used By SpineHealth?
──────┼────────────────┼─────────────────────
  0   │ Nose           │ ✅ Head centroid
  1   │ Left Eye       │ ❌ (not used)
  2   │ Right Eye      │ ❌ (not used)
  3   │ Left Ear       │ ✅ Head centroid + fallback scale
  4   │ Right Ear      │ ✅ Head centroid + fallback scale
  5   │ Left Shoulder  │ ✅ Shoulder centroid + primary scale
  6   │ Right Shoulder │ ✅ Shoulder centroid + primary scale
  7   │ Left Elbow     │ ❌
  8   │ Right Elbow    │ ❌
  9   │ Left Wrist     │ ❌
 10   │ Right Wrist    │ ❌
 11   │ Left Hip       │ (defined, reserved for future use)
 12   │ Right Hip      │ (defined, reserved for future use)
 13   │ Left Knee      │ ❌
 14   │ Right Knee     │ ❌
 15   │ Left Ankle     │ ❌
 16   │ Right Ankle    │ ❌
```

### NCNN Model Conversion
The standard YOLOv8n-pose `.pt` file must be exported to NCNN format for optimized ARM execution. This is done once:
```bash
# On any machine with Ultralytics installed:
yolo export model=yolov8n-pose.pt format=ncnn
# This generates: yolov8n-pose_ncnn_model/ directory
```
The resulting `yolov8n-pose_ncnn_model/` folder is then copied to the Pi and referenced in `main.py`.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Raspberry Pi 5                           │
│                                                                 │
│  IMX219 Camera                                                  │
│       │                                                         │
│       ▼                                                         │
│  ┌─────────────┐    read()       ┌──────────────────────────┐   │
│  │  OpenCV     │──────────────▶ │  YOLOv8n-pose (NCNN)     |    |
│  │  V4L2 Cap   │                 │  ARM-optimized inference │   │
│  │  640×480    │                 └────────────┬─────────────┘   │
│  └─────────────┘                             │                  │
│                                              │ 17 keypoints     │
│                                              │ (x, y, conf)     │
│                                             ▼                   │
│                              ┌──────────────────────────────┐   │
│                              │  compute_posture_index()      │  │
│                              │                               │  │
│                              │  1. Weighted Head Centroid    │  │
│                              │  2. Shoulder Centroid         │  │
│                              │  3. d_v (vertical gap)        │  │
│                              │  4. d_h (shoulder width)      │  │
│                              │     + EMA fallback chain      │  │
│                              │  5. NPR = d_v / d_h           │  │
│                              └──────────────┬───────────────┘   │
│                                             │                   │
│                               NPR < 0.45?  │                    │
│                              ┌──────────────┼──────────────┐    │
│                              │              │              │    │
│                         SLOUCH           GOOD POSTURE      │    │
│                              │              │              │    │
│                    ┌─────────▼──┐    ┌──────▼──────┐       │    │
│                    │ Timer > 5s?│    │ Green LED ON │       │    │
│                    └─────────┬──┘    └─────────────┘       │    │
│                              │                              │   │
│                    ┌─────────▼──┐                          │   │
│                    │ Red LED ON │                          │   │
│                    │ Log Event  │                          │   │
│                    └────────────┘                          │   │
│                                                             │   │
│              ┌─────────────────────────────────────────┐   │   │
│              │  SQLite: workspace_metrics.db            │   │   │
│              │  events: DESK_ARRIVAL / DESK_DEPARTURE   │   │   │
│              │           / SLOUCH_DETECTED              │   │   │
│              └─────────────────────────────────────────┘   │   │
└─────────────────────────────────────────────────────────────────┘
```

---

## The Mathematical Core — Normalized Posture Ratio (NPR)

### The Problem with Simple 2D Angles
A naive approach would measure the angle between the nose, mid-shoulder point, and some horizontal reference. However, with a **front-facing desk camera**, a user who leans slightly forward or backward changes their apparent size in the frame. This makes raw pixel distances unreliable — a tall person sitting upright and a short person slouching could yield the same raw pixel distances.

### The Solution: Scale-Normalized Vertical Ratio

SpineHealth defines the **Normalized Posture Ratio (NPR)**:

```
         d_v   (Head Centroid Y − Shoulder Centroid Y) in pixels
NPR  =  ─── = ─────────────────────────────────────────────────────
         d_h            Bi-acromial Width in pixels
```

Where:
- **`d_v` (Vertical Extension):** The vertical pixel distance from the shoulder centroid to the head centroid. Since Y increases downward in image space, `d_v = Shoulder_Y − Head_Y`. A good, upright posture yields a **larger** `d_v` (head is far above shoulders). Slouching shrinks `d_v` (head drops toward chest level).
- **`d_h` (Bi-acromial Width):** The pixel distance between the left and right shoulders. This acts as a **person-specific, distance-adaptive scale**. If the user moves closer to the camera, both `d_v` and `d_h` grow proportionally, keeping the ratio stable.

#### Why This Works
- **Scale-invariant:** Dividing by shoulder width normalizes for camera-to-person distance.
- **Person-agnostic:** Different torso lengths don't matter because each person's own shoulder width is used as the reference.
- **Perspective-robust:** Suitable for cameras placed directly in front of the user (desk/monitor mount).

### Confidence-Weighted Centroids

Raw keypoints are noisy. SpineHealth computes a **confidence-weighted centroid** rather than using any single point:

```python
# For the Head Centroid:
Head_C = weighted_average([Nose, Left_Ear, Right_Ear])

# For the Shoulder Centroid:
Shoulder_C = weighted_average([Left_Shoulder, Right_Shoulder])
```

Each keypoint's contribution is weighted by its detection confidence score. Low-confidence detections (below `CONF_THRESHOLD = 0.35`) are zeroed out via a **gating function**, preventing unreliable detections from polluting the centroid calculation.

### The 3-Tier Fallback Chain for `d_h`

Shoulder width can momentarily become undetectable (e.g., arms raised, occlusion). Without a fallback, the NPR computation would fail entirely. SpineHealth uses a **3-tier fallback chain**:

```
Tier 1 (Full Confidence):  Both shoulders visible → Use live pixel distance
                                    ↓ (if fails)
Tier 2 (EMA Fallback):     Use the Exponential Moving Average of recent shoulder widths
                            Reliability penalty: 0.8×
                                    ↓ (if fails)
Tier 3 (Cold-Start):       Use ear-to-ear width × 2.1 (anthropometric constant)
                            Reliability penalty: 0.5×
```

The **Exponential Moving Average (EMA)** for shoulder width is maintained with `α = 0.1`:
```
EMA_new = (0.1 × live_width) + (0.9 × EMA_old)
```
This smooths micro-jitters in detection across frames while keeping the estimate current.

The **anthropometric constant of 2.1** is derived from human anatomy studies: the average shoulder width is approximately 2.1× the head width (ear-to-ear distance). This provides a reasonable cold-start estimate before the EMA has been populated.

### Reliability Score

Every `PostureResult` includes a **reliability score** (0.0–1.0) calculated as:
```
reliability = min(1.0, head_centroid_weight / 3.0) × fallback_penalty
```

This gives downstream logic the ability to ignore or soft-weight measurements when keypoint detection quality is poor (e.g., user is partially out of frame).

---

## Code Walkthrough

### 1. Database Initialization

```python
def init_db():
    conn = sqlite3.connect("workspace_metrics.db")
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            event_type TEXT,
            value REAL
        )
    ''')
    conn.commit()
    return conn
```

**`init_db()`** creates (or connects to) `workspace_metrics.db` in the current working directory. It uses `CREATE TABLE IF NOT EXISTS` so it is safely idempotent — running the script multiple times won't destroy previous logs.

**`log_event(conn, event_type, value=0.0)`** inserts a single row using a parameterized query (`?` placeholders), which is safe against SQL injection even though the data source is internal.

---

### 2. Posture Math & Keypoint Processing

#### `PostureResult` NamedTuple
```python
class PostureResult(NamedTuple):
    posture_ratio: Optional[float]   # The NPR value (None if uncomputable)
    vertical_ext:  Optional[float]   # Raw d_v in pixels
    shoulder_width: Optional[float]  # d_h used (live, EMA, or estimated)
    reliability: float               # 0.0 to 1.0 confidence score
```

Using a `NamedTuple` instead of a plain tuple ensures the return value is self-documenting, immutable, and efficiently memory-laid-out.

#### `_gated_weight(conf, threshold=0.35)`
Returns the confidence score if it is above the threshold, otherwise `0.0`. This hard gate prevents near-zero confidence keypoints (which may be random noise from the model) from pulling the centroid in wrong directions.

#### `_weighted_centroid(points)`
Takes a list of `(x, y, confidence)` tuples. Returns the confidence-weighted average `(x, y)` position and the total weight sum. If all points fall below the confidence threshold (total weight ≈ 0), returns `(None, 0.0)` to signal a failed detection.

#### `compute_posture_index(keypoints, state)`
The main algorithm function. Accepts:
- `keypoints`: a `(17, 3)` NumPy array from YOLO (x, y, confidence per landmark).
- `state`: a mutable dict used to persist the EMA across frames.

Returns a `PostureResult`.

---

### 3. Edge Execution Pipeline

The `main()` function orchestrates the full system:

#### Initialization Phase
```python
conn = init_db()          # Open/create the SQLite database
red_led = LED(17)         # BCM pin 17 → Physical pin 11
green_led = LED(27)       # BCM pin 27 → Physical pin 13
model = YOLO("yolov8n-pose_ncnn_model")  # Load NCNN-optimized model
cap = cv2.VideoCapture(0) # Open Pi Camera via V4L2 driver
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
```

#### Main Loop
The loop reads camera frames at the native camera frame rate but **only processes every 5th frame** (`FRAME_SKIP = 5`). This reduces CPU usage to ~20% of full processing load while still delivering ~6 analyzed frames per second (at 30 FPS camera input).

```python
while cap.isOpened():
    ret, frame = cap.read()
    frame_count += 1

    if frame_count % FRAME_SKIP != 0:
        continue  # Skip this frame entirely

    results = model(frame, verbose=False)  # Run NCNN inference
    # → extract keypoints → compute NPR → update LEDs → log events
```

#### Resource Cleanup (Finally Block)
```python
finally:
    cap.release()       # Release the camera resource
    conn.close()        # Flush and close the database
    red_led.off()       # Ensure LEDs are off when script exits
    green_led.off()
```
The `finally` block guarantees hardware and database resources are properly released even if an exception or `KeyboardInterrupt` occurs.

---

## State Machines

SpineHealth implements two independent state machines running concurrently in the main loop:

### Posture State Machine

```
                    ┌──────────────────────────────────────────┐
                    │              Each Processed Frame         │
                    └─────────────────┬────────────────────────┘
                                      │
                         ┌────────────▼────────────┐
                         │  NPR < SLOUCH_THRESHOLD? │
                         └──────────┬──────────────┘
                                    │
              ┌─────────────────────┼─────────────────────────┐
              │ YES (slouching)      │                          │ NO (good posture)
              ▼                     │                          ▼
    ┌─────────────────┐             │             ┌─────────────────────┐
    │ Green LED OFF   │             │             │ Red LED OFF         │
    │ Start timer if  │             │             │ Green LED ON        │
    │ not started     │             │             │ Reset slouch timer  │
    └────────┬────────┘             │             └─────────────────────┘
             │                      │
    Timer > SLOUCH_TIME_THRESHOLD?  │
             │                      │
    ┌────────▼────────┐             │
    │ Red LED ON      │             │
    │ Log SLOUCH_     │             │
    │ DETECTED event  │             │
    │ Reset timer     │             │
    └─────────────────┘             │
```

**Key design detail:** After logging a `SLOUCH_DETECTED` event, the timer resets to the current time. This prevents the database from being flooded with thousands of rows for a single slouching session — it will log once per `SLOUCH_TIME_THRESHOLD` interval (default: every 5 seconds of sustained slouching).

### Occupancy State Machine

```
                    Each Frame
                        │
          ┌─────────────▼─────────────┐
          │  Person detected?          │
          └──────────┬────────────────┘
                     │
    ┌────────────────┼────────────────────┐
    │ YES             │                   │ NO
    ▼                 │                   ▼
 Not present          │          Still was present?
 previously?          │                   │
    │                 │    ┌──────────────┼──────────────┐
    ▼                 │    │ YES          │              │ NO
 Log DESK_ARRIVAL     │    ▼              │              ▼
 is_present = True    │  Time since      │         Do nothing
 Update last_seen     │  last_seen >     │
                      │  BREAK_THRESHOLD?│
                      │    │             │
                      │    ▼             │
                      │  Log DESK_       │
                      │  DEPARTURE       │
                      │  is_present=False│
                      │  LEDs OFF        │
```

**Key design detail:** The system waits `BREAK_TIME_THRESHOLD` (default: 60 seconds) of continuous absence before logging a departure. This prevents false departure events for brief moments when YOLO misses a detection (e.g., user bends down to pick something up).

---

## Configuration & Tuning

All thresholds are defined as constants near the top of `main()` for easy tuning:

```python
SLOUCH_RATIO_THRESHOLD = 0.45   # NPR below this → slouching
SLOUCH_TIME_THRESHOLD  = 5      # Seconds of slouching before alert
BREAK_TIME_THRESHOLD   = 60     # Seconds absent → desk departure
FRAME_SKIP             = 5      # Process 1 in every N frames
CONF_THRESHOLD         = 0.35   # Minimum keypoint confidence (global)
```

| Parameter | Effect of Increasing | Effect of Decreasing |
|---|---|---|
| `SLOUCH_RATIO_THRESHOLD` | More sensitive (triggers on milder slouches) | Less sensitive (only extreme slouches trigger) |
| `SLOUCH_TIME_THRESHOLD` | Longer sustained slouch needed for alert | Alert fires faster |
| `BREAK_TIME_THRESHOLD` | Longer absence needed before logging departure | More sensitive to brief absences |
| `FRAME_SKIP` | Lower CPU usage, lower detection rate | Higher CPU usage, more responsive |
| `CONF_THRESHOLD` | Fewer, higher-quality keypoints used | More keypoints used (including noisy ones) |

---

## Privacy Guarantee

The privacy guarantee is enforced structurally — not by policy:

1. **No display functions:** `cv2.imshow()` is never called.
2. **No write functions:** `cv2.imwrite()`, `cv2.VideoWriter`, and similar are never called.
3. **No network calls:** There are no sockets, HTTP clients, or cloud SDKs in the codebase.
4. **Frame lifecycle:** Each frame is read from the camera into a local variable named `frame`. After YOLO processes it and keypoints are extracted, the loop immediately iterates — Python's garbage collector reclaims the frame's memory. It is never held in a buffer, queue, or persistent variable.
5. **Database content:** The only data stored is:
   - A timestamp (when the event occurred)
   - An event type string (`"SLOUCH_DETECTED"`, `"DESK_ARRIVAL"`, `"DESK_DEPARTURE"`)
   - A single float (the NPR ratio at the time of the event)

No faces, no images, no video — ever.

---

## 🛠 Installation & Setup

### Prerequisites

```bash
# Update the Pi OS
sudo apt update && sudo apt upgrade -y

# Install system dependencies for OpenCV
sudo apt install -y python3-pip python3-dev libopencv-dev python3-opencv

# Enable the camera
sudo raspi-config
# → Interface Options → Camera → Enable
```

### Python Dependencies

```bash
# Install Ultralytics (includes YOLO and NCNN export support)
pip3 install ultralytics

# Install gpiozero (usually pre-installed on Pi OS)
pip3 install gpiozero

# NumPy and OpenCV Python bindings
pip3 install numpy opencv-python
```

### Prepare the NCNN Model

Exporting to NCNN format can be done on any machine (not necessarily the Pi), then the output folder is transferred:

```bash
# On a development machine (with GPU/faster CPU):
pip install ultralytics
python3 -c "from ultralytics import YOLO; YOLO('yolov8n-pose.pt').export(format='ncnn')"

# Transfer the generated folder to the Pi:
scp -r yolov8n-pose_ncnn_model/ pi@<PI_IP>:~/SpineHealth/
```

### Clone & Place Files

```bash
# On the Raspberry Pi:
git clone https://github.com/archit2k5/SpineHealth.git
cd SpineHealth
# Ensure yolov8n-pose_ncnn_model/ is in this directory
```

---

## Running the Monitor

```bash
cd ~/SpineHealth

# Run the monitor (requires sudo for GPIO access on some Pi OS versions)
sudo python3 main.py
```

**Expected startup output:**
```
Loading optimized NCNN model...
Pipeline started securely. Press Ctrl+C to stop.
```

**Example runtime output:**
```
Logged: DESK_ARRIVAL | Value: 0.00
Logged: SLOUCH_DETECTED | Value: 0.38
Logged: DESK_DEPARTURE | Value: 0.00
```

**To stop:**
```
Ctrl+C
```
The `finally` block will cleanly release the camera and close the database.

### Auto-Start on Boot (Optional)

To run SpineHealth automatically when the Pi powers on:

```bash
# Create a systemd service
sudo nano /etc/systemd/system/spinehealth.service
```

```ini
[Unit]
Description=SpineHealth Posture Monitor
After=multi-user.target

[Service]
ExecStart=/usr/bin/python3 /home/pi/SpineHealth/main.py
WorkingDirectory=/home/pi/SpineHealth
Restart=on-failure
User=root

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable spinehealth
sudo systemctl start spinehealth
```

---

## Database Schema & Event Logging

The SQLite database `workspace_metrics.db` is created in the working directory. To inspect it:

```bash
sqlite3 workspace_metrics.db
```

```sql
-- View all events
SELECT * FROM events ORDER BY timestamp DESC;

-- Count today's slouch events
SELECT COUNT(*) FROM events
WHERE event_type = 'SLOUCH_DETECTED'
AND date(timestamp) = date('now');

-- View session summary (arrival/departure pairs)
SELECT * FROM events
WHERE event_type IN ('DESK_ARRIVAL', 'DESK_DEPARTURE')
ORDER BY timestamp;

-- Average NPR when slouching was detected
SELECT AVG(value) FROM events WHERE event_type = 'SLOUCH_DETECTED';
```

**Schema:**

| Column | Type | Description |
|---|---|---|
| `id` | INTEGER | Auto-incrementing primary key |
| `timestamp` | DATETIME | UTC timestamp of the event (auto-set by SQLite) |
| `event_type` | TEXT | `DESK_ARRIVAL`, `DESK_DEPARTURE`, or `SLOUCH_DETECTED` |
| `value` | REAL | NPR ratio for slouch events; `0.0` for occupancy events |

---

## Performance Optimizations

| Optimization | Implementation | Effect |
|---|---|---|
| **NCNN Runtime** | Model exported to NCNN format | ~3× faster inference vs. PyTorch on ARM CPU |
| **Low Resolution Capture** | 640×480 vs. native 1080p | ~4× reduction in pixel data processed per frame |
| **Frame Skipping** | Process 1 of every 5 frames | ~80% reduction in inference calls |
| **`verbose=False`** | Suppresses YOLO console output | Eliminates string formatting overhead |
| **EMA for Shoulder Width** | Persistent state across frames | Avoids redundant full shoulder re-computation |
| **NumPy Vectorization** | Centroid calculation uses array ops | Faster than Python loops for multi-point averaging |

**Estimated CPU utilization on Pi 5:** ~25–35% on a single core at FRAME_SKIP=5.

---

## Calibration Guide

Because human body proportions and desk setups vary, you should calibrate `SLOUCH_RATIO_THRESHOLD` for your specific setup:

### Step 1: Dry-Run in Ratio-Logging Mode

Temporarily add a print statement to see live NPR values:

```python
if posture_res.posture_ratio is not None:
    print(f"NPR: {posture_res.posture_ratio:.3f} | Reliability: {posture_res.reliability:.2f}")
```

### Step 2: Measure Your Personal Baselines

1. Sit in **your best, most upright posture** for 30 seconds. Note the average NPR (expect ~0.55–0.80).
2. Sit in **your typical slouch** for 30 seconds. Note the average NPR (expect ~0.30–0.50).

### Step 3: Set Your Threshold

Set `SLOUCH_RATIO_THRESHOLD` to approximately halfway between your two baseline values. For example, if upright = 0.70 and slouch = 0.40, set the threshold to `0.55`.

### Step 4: Adjust Timing

If alerts are too frequent during natural movement:
- Increase `SLOUCH_TIME_THRESHOLD` from 5 to 10–15 seconds.

If alerts feel delayed:
- Decrease `SLOUCH_TIME_THRESHOLD` to 3 seconds.

---

## Troubleshooting

| Issue | Likely Cause | Solution |
|---|---|---|
| `camera not found` | Camera not enabled or connected | Run `sudo raspi-config` → enable Camera interface; check CSI ribbon cable |
| `GPIO not found` | `gpiozero` not installed or wrong permissions | `pip install gpiozero`; run with `sudo` |
| `NCNN model not found` | Model folder missing | Export NCNN model and place `yolov8n-pose_ncnn_model/` in project root |
| No LED response | Wiring issue or wrong BCM pin numbers | Double-check wiring against the diagram; verify `LED(17)` matches physical pin 11 |
| NPR always `None` | User out of frame or low confidence | Move closer to camera; ensure good lighting; check `CONF_THRESHOLD` |
| Database not updating | Permissions issue | Check write permissions on the working directory |
| High CPU / thermal throttling | Too many frames processed | Increase `FRAME_SKIP` to 8 or 10 |

---

## Project Structure

```
SpineHealth/
│
├── main.py                        # Core application — all logic
├── workspace_metrics.db           # SQLite event log (auto-created on first run)
├── yolov8n-pose_ncnn_model/       # NCNN-optimized model (add before running)
│   ├── model.ncnn.bin
│   └── model.ncnn.param
└── README.md                      # This file
```

---

## Future Roadmap

- [ ] **Web dashboard** — Flask/FastAPI endpoint to visualize event history from `workspace_metrics.db` in a browser.
- [ ] **Multi-person support** — process multiple detected skeletons and associate events per-person via bounding box tracking.
- [ ] **Lateral lean detection** — extend NPR with a horizontal symmetry ratio between left/right keypoints.
- [ ] **Ergonomic session reports** — daily/weekly summary exports (CSV or PDF) generated from the database.
- [ ] **Buzzer feedback** — add an active buzzer on a third GPIO pin for an auditory alert on slouch.
- [ ] **Pi Camera 3 support** — upgrade to the 12MP IMX708 sensor for improved low-light performance.

*SpineHealth — because your spine shouldn't be an afterthought.*

