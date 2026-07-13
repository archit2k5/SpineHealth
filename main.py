"""
Privacy-First Posture & Occupancy Edge Monitor
Hardware: Raspberry Pi 5, IMX219 Camera, Red/Green LEDs
Model: YOLOv8n-pose (NCNN compiled for ARM CPU optimization)

This script calculates a Normalized Posture Ratio (NPR) from a front-facing
camera feed. It provides real-time LED feedback and logs events to a local SQLite
database. No video frames are ever saved or transmitted.
"""

import cv2
import math
import time
import sqlite3
import numpy as np
from typing import Optional, Tuple, NamedTuple
from ultralytics import YOLO
from gpiozero import LED

# ==========================================
# 1. Database Initialization
# ==========================================
def init_db():
    """Initializes the local SQLite database for event logging."""
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

def log_event(conn, event_type, value=0.0):
    """Logs posture or occupancy events to the local database."""
    cursor = conn.cursor()
    cursor.execute("INSERT INTO events (event_type, value) VALUES (?, ?)", (event_type, value))
    conn.commit()
    print(f"Logged: {event_type} | Value: {value:.2f}")

# ==========================================
# 2. Robust Posture Math (Normalized Ratio)
# ==========================================
NOSE, L_EYE, R_EYE, L_EAR, R_EAR = 0, 1, 2, 3, 4
L_SHOULDER, R_SHOULDER = 5, 6
L_HIP, R_HIP = 11, 12

CONF_THRESHOLD = 0.35      

class PostureResult(NamedTuple):
    posture_ratio: Optional[float]   
    vertical_ext: Optional[float]    
    shoulder_width: Optional[float]  
    reliability: float               

def _gated_weight(conf: float, threshold: float = CONF_THRESHOLD) -> float:
    return float(conf) if conf >= threshold else 0.0

def _weighted_centroid(points: list[Tuple[float, float, float]]) -> Tuple[Optional[np.ndarray], float]:
    coords = np.array([[p[0], p[1]] for p in points], dtype=np.float64)
    weights = np.array([_gated_weight(p[2]) for p in points], dtype=np.float64)
    total_weight = weights.sum()

    if total_weight <= 1e-6:
        return None, 0.0

    centroid = (coords * weights[:, None]).sum(axis=0) / total_weight
    return centroid, float(total_weight)

def compute_posture_index(keypoints: np.ndarray, state: dict) -> PostureResult:
    """
    Computes a Scale-Normalized Vertical Ratio (R) with a robust fallback chain 
    for the horizontal shoulder reference (d_h).
    """
    if keypoints is None or keypoints.shape[0] < 17:
        return PostureResult(None, None, None, 0.0)

    kp = keypoints

    # 1. Build the Head Centroid (H)
    head_pts = [tuple(kp[NOSE]), tuple(kp[L_EAR]), tuple(kp[R_EAR])]
    head_c, w_head = _weighted_centroid(head_pts)

    # 2. Build the Shoulder Centroid (S)
    shoulder_pts = [tuple(kp[L_SHOULDER]), tuple(kp[R_SHOULDER])]
    shoulder_c, w_shoulder = _weighted_centroid(shoulder_pts)

    if head_c is None or shoulder_c is None:
        return PostureResult(None, None, None, 0.0)

    # 3. Calculate Vertical Gap (d_v)
    d_v = float(shoulder_c[1] - head_c[1])
    if d_v <= 0:
        return PostureResult(None, d_v, None, 0.0)

    # 4. Calculate Horizontal Scale (d_h) with Fallback Chain
    l_sh_conf = kp[L_SHOULDER][2]
    r_sh_conf = kp[R_SHOULDER][2]
    
    d_h = None
    reliability_penalty = 1.0

    # Primary: Live shoulder-to-shoulder distance
    if l_sh_conf >= CONF_THRESHOLD and r_sh_conf >= CONF_THRESHOLD:
        live_d_h = math.sqrt(
            (kp[L_SHOULDER][0] - kp[R_SHOULDER][0])**2 + 
            (kp[L_SHOULDER][1] - kp[R_SHOULDER][1])**2
        )
        
        # Update Exponential Moving Average (EMA)
        # alpha = 0.1 means we trust the history 90% and the new frame 10%
        # This smooths out micro-jitters in the bounding box
        alpha = 0.1 
        if state.get('shoulder_ema') is None:
            state['shoulder_ema'] = live_d_h
        else:
            state['shoulder_ema'] = (alpha * live_d_h) + ((1 - alpha) * state['shoulder_ema'])
            
        d_h = live_d_h

    # Fallback 1: Exponential Moving Average (EMA)
    elif state.get('shoulder_ema') is not None:
        d_h = state['shoulder_ema']
        reliability_penalty = 0.8 # Slightly reduce confidence since we are using old data

    # Fallback 2: Cold-Start Anthropometric Constant (Ears * 2.1)
    else:
        l_ear_conf = kp[L_EAR][2]
        r_ear_conf = kp[R_EAR][2]
        
        if l_ear_conf >= CONF_THRESHOLD and r_ear_conf >= CONF_THRESHOLD:
            ear_width = math.sqrt(
                (kp[L_EAR][0] - kp[R_EAR][0])**2 + 
                (kp[L_EAR][1] - kp[R_EAR][1])**2
            )
            d_h = ear_width * 2.1
            reliability_penalty = 0.5 # Lowest confidence, relying on biological estimates

    # If all fallbacks fail, we cannot compute the ratio
    if d_h is None or d_h <= 1e-6:
        return PostureResult(None, d_v, d_h, 0.0)

    # 5. Compute the Final Ratio (R)
    posture_ratio = d_v / d_h

    # Base reliability on head tracking, penalized by the shoulder fallback level
    max_possible_head_w = 3.0 
    reliability = min(1.0, (w_head / max_possible_head_w)) * reliability_penalty

    return PostureResult(posture_ratio, d_v, d_h, reliability)
# ==========================================
# 3. Edge Execution Pipeline
# ==========================================
def main():
    conn = init_db()
    
    # Initialize GPIO LEDs
    red_led = LED(17)
    green_led = LED(27)
    
    # Load the optimized NCNN model for Raspberry Pi CPU execution
    print("Loading optimized NCNN model...")
    model = YOLO("yolov8n-pose_ncnn_model")
    
    # Initialize Pi Camera (0) at low resolution for faster processing
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    is_present = False
    last_seen_time = time.time()
    slouch_start_time = None
    frame_count = 0
    
    # --- CONFIGURATION THRESHOLDS ---
    # Adjust this based on your PC calibration tests (e.g., 0.45)
    SLOUCH_RATIO_THRESHOLD = 0.45 
    
    SLOUCH_TIME_THRESHOLD = 5     # Seconds of slouching before logging/Red LED
    BREAK_TIME_THRESHOLD = 60     # Seconds before marking desk as empty
    FRAME_SKIP = 5                # Process 1 out of every 5 frames (~6 FPS)
    
    print("Pipeline started securely. Press Ctrl+C to stop.")
    
    try:
        # Ensure LEDs are off at boot
        red_led.off()
        green_led.off()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            frame_count += 1
            
            # Optimization: Skip frames to save CPU and maintain zero latency
            if frame_count % FRAME_SKIP != 0:
                continue
                
            results = model(frame, verbose=False)
            person_detected_this_frame = False

            for result in results:
                keypoints = result.keypoints
                
                # Verify that a person was detected and array has valid shape
                if keypoints is not None and keypoints.data.shape[0] > 0:
                    person_detected_this_frame = True
                    
                    kp_data = keypoints.data[0].cpu().numpy()
                    posture_res = compute_posture_index(kp_data)
                    
                    if posture_res.posture_ratio is not None:
                        current_ratio = posture_res.posture_ratio
                        
                        # Slouching State Machine
                        if current_ratio < SLOUCH_RATIO_THRESHOLD:
                            # User is slouching
                            green_led.off()
                            
                            if slouch_start_time is None:
                                slouch_start_time = time.time()
                            
                            # Trigger Red LED and log event if threshold time is passed
                            if (time.time() - slouch_start_time) > SLOUCH_TIME_THRESHOLD:
                                red_led.on()
                                # Prevent log spamming by resetting the timer
                                log_event(conn, "SLOUCH_DETECTED", current_ratio)
                                slouch_start_time = time.time() 
                        else:
                            # Good posture
                            slouch_start_time = None 
                            red_led.off()
                            green_led.on()
            
            # Occupancy State Machine
            current_time = time.time()
            if person_detected_this_frame:
                if not is_present:
                    log_event(conn, "DESK_ARRIVAL")
                    is_present = True
                last_seen_time = current_time
            else:
                if is_present and (current_time - last_seen_time) > BREAK_TIME_THRESHOLD:
                    log_event(conn, "DESK_DEPARTURE")
                    is_present = False
                    slouch_start_time = None
                    
                # Turn off LEDs when no one is at the desk
                if not is_present:
                    red_led.off()
                    green_led.off()

            # PRIVACY: No visual output functions (cv2.imshow/imwrite) are called.
            # The 'frame' array is safely discarded and overwritten on the next loop.

    except KeyboardInterrupt:
        print("\nShutting down pipeline safely...")
    except Exception as e:
        print(f"\nCritical Error: {e}")
    finally:
        # Hardware & Resource Cleanup
        cap.release()
        conn.close()
        red_led.off()
        green_led.off()

if __name__ == "__main__":
    main()