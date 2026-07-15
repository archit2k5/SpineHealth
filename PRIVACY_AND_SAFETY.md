# Privacy and Safety

## Data handling
Only three fields are ever stored: timestamp, event type (`DESK_ARRIVAL`/`DESK_DEPARTURE`/`SLOUCH_DETECTED`), and a single NPR float value. No images, video frames, or raw keypoints are saved.

## Permissions
Camera access only (via Picamera2/libcamera). GPIO access for LED output.

## Storage
Local SQLite file (`workspace_metrics.db`) on the device's own storage. No cloud sync, no external transmission.

## Limitations
- Single-person tracking only; behavior with multiple people in frame is undefined.
- Posture thresholds are heuristic (manually calibrated), not clinically validated — not a medical device.
- NPR can miss whole-torso lean when the neck/back segment stays locally straight (see Evaluation).

## Risks
- Even without storing images, continuous presence/posture logging is monitoring-adjacent; deployment in shared/workplace settings should include user awareness/consent.
