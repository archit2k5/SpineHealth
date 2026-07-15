# Local AI Verification

## Runs fully on-device
- Camera capture (Picamera2)
- Pose inference (YOLOv8n-pose, NCNN, ARM CPU)
- Posture/occupancy logic
- Event logging (SQLite, local file)
- LED feedback (GPIO)

## Requires internet
None. No network calls exist in the codebase.

## Data leaving the device
None. No images, video, or keypoints are transmitted or persisted — only aggregated numeric event logs (timestamp, event type, NPR float value) are written to a local SQLite file.

## Note on debug preview
When `DEBUG_DISPLAY = True`, an annotated video window is shown locally on-screen for demonstration purposes. This is a local display only — it is not saved to disk or transmitted. In headless mode (`DEBUG_DISPLAY = False`), no display is rendered at all.
