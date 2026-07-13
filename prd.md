# Product Requirement Document (PRD)

## 1. Project Overview
* **Product Name:** Privacy-First Posture & Tardiness Detector
* **Target Audience:** Remote workers, privacy-conscious enterprises, physical wellness enthusiasts.
* **Core Value Proposition:** Real-time workplace wellness and attendance monitoring powered strictly on edge hardware, ensuring zero video frames ever leave the local network.

---

## 2. System Hardware & Deployment Architecture
* **Primary Processor:** Raspberry Pi 5 (8GB recommended for fluid frame processing).
* **Sensor Input:** Waveshare IMX219 Camera Module (CSI interface).
* **Edge Inference Engine:** MediaPipe Pose / YOLOv8-Pose (quantized ONNX or TFLite compiled for ARM64).
* **Local Storage:** SQLite (for lightweight event logging).
* **Frontend Dashboard:** Local Web UI served over HTTP/WebSockets (`http://localhost:5001`).

---

## 3. Key Features & Functional Requirements

### 3.1 Local Posture Analysis
* **Mechanism:** Tracks key body landmarks (ears, shoulders, hips) to compute cervical spine angle and slouching threshold.
* **Events Detected:** Slouching, forward head posture, asymmetric shoulder alignment.
* **Alert Trigger:** Audio ping or local web push notification if poor posture persists for > 2 minutes.

### 3.2 Occupancy & Prolonged Sitting Tracker
* **Mechanism:** Pose detection confidence on desk boundary zone.
* **Events Detected:** `USER_PRESENT`, `USER_ABSENT`, `PROLONGED_SITTING`.
* **Break Alert:** Triggers a "Time to stretch" recommendation after 60 continuous minutes of seated activity.

### 3.3 Privacy-Preserving Tardiness Tracker
* **Mechanism:** Monitors first `USER_PRESENT` timestamp after configured workday start time (e.g., 09:00 AM).
* **Data Logged:** Timestamp string and late status boolean (e.g., `2026-07-14 09:18:22 | TARDY`). No image frames are captured or persisted.

---

## 4. Non-Functional Requirements & Security
* **Zero Cloud Streaming:** Camera frames must process directly in RAM via Video4Linux2 / `libcamera` and be discarded immediately.
* **Performance:** Minimum 15 FPS at 720p stream resolution on Pi 5 without thermal throttling.
* **Data Privacy:** SQLite database stores strictly numerical event logs (timestamps, angles, duration).

---

## 5. Technical Specifications & Metrics
* **Posture Threshold:** Spine inclination angle $> 25^\circ$ off vertical axis = Slouch.
* **Latency:** Notification triggering within $< 500\text{ ms}$ of rule breach.