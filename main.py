"""
Privacy-First Posture & Occupancy Edge Monitor
Hardware: Raspberry Pi 5, IMX219 Camera, Red/Green LEDs
Model: YOLOv8n-pose (NCNN compiled for ARM CPU optimization)
Camera Backend: Picamera2 (libcamera native)
"""

import cv2  # Re-imported for debug display
import math
import time
import sqlite3
import numpy as np
from typing import Optional, Tuple, NamedTuple
from ultralytics import YOLO
from gpiozero import LED
from picamera2 import Picamera2

# ==========================================
# 1. Database Initialization
# ==========================================
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

def log_event(conn, event_type, value=0.0):
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
    if keypoints is None or keypoints.shape[0] < 17:
        return PostureResult(None, None, None, 0.0)

    kp = keypoints

    head_pts = [tuple(kp[NOSE]), tuple(kp[L_EAR]), tuple(kp[R_EAR])]
    head_c, w_head = _weighted_centroid(head_pts)

    shoulder_pts = [tuple(kp[L_SHOULDER]), tuple(kp[R_SHOULDER])]
    shoulder_c, w_shoulder = _weighted_centroid(shoulder_pts)

    if head_c is None or shoulder_c is None:
        return PostureResult(None, None, None, 0.0)

    d_v = float(shoulder_c[1] - head_c[1])
    if d_v <= 0:
        return PostureResult(None, d_v, None, 0.0)

    l_sh_conf = kp[L_SHOULDER][2]
    r_sh_conf = kp[R_SHOULDER][2]
    
    d_h = None
    reliability_penalty = 1.0

    if l_sh_conf >= CONF_THRESHOLD and r_sh_conf >= CONF_THRESHOLD:
        live_d_h = math.sqrt((kp[L_SHOULDER][0] - kp[R_SHOULDER][0])**2 + (kp[L_SHOULDER][1] - kp[R_SHOULDER][1])**2)
        alpha = 0.1 
        if state.get('shoulder_ema') is None:
            state['shoulder_ema'] = live_d_h
        else:
            state['shoulder_ema'] = (alpha * live_d_h) + ((1 - alpha) * state['shoulder_ema'])
        d_h = live_d_h

    elif state.get('shoulder_ema') is not None:
        d_h = state['shoulder_ema']
        reliability_penalty = 0.8 
    else:
        l_ear_conf = kp[L_EAR][2]
        r_ear_conf = kp[R_EAR][2]
        
        if l_ear_conf >= CONF_THRESHOLD and r_ear_conf >= CONF_THRESHOLD:
            ear_width = math.sqrt((kp[L_EAR][0] - kp[R_EAR][0])**2 + (kp[L_EAR][1] - kp[R_EAR][1])**2)
            d_h = ear_width * 2.1
            reliability_penalty = 0.5 

    if d_h is None or d_h <= 1e-6:
        return PostureResult(None, d_v, d_h, 0.0)

    posture_ratio = d_v / d_h
    reliability = min(1.0, (w_head / 3.0)) * reliability_penalty

    return PostureResult(posture_ratio, d_v, d_h, reliability)

# ==========================================
# 3. Edge Execution Pipeline (Picamera2)
# ==========================================
def main():
    conn = init_db()
    red_led = LED(17)
    green_led = LED(27)
    
    # --- DEBUG TOGGLE ---
    # Set to False when you want the Pi to run securely and invisibly in the background.
    DEBUG_DISPLAY = True 

    print("Loading optimized NCNN model...")
    model = YOLO("yolov8n-pose_ncnn_model")
    
    print("Starting IMX219 via Picamera2...")
    picam2 = Picamera2()
    config = picam2.create_video_configuration(main={"size": (1280, 720), "format": "RGB888"})
    picam2.configure(config)
    picam2.start()
    
    is_present = False
    last_seen_time = time.time()
    slouch_start_time = None
    frame_count = 0
    tracker_state = {} 
    
    SLOUCH_RATIO_THRESHOLD = 0.6
    SLOUCH_TIME_THRESHOLD = 5   
    BREAK_TIME_THRESHOLD = 60     
    
    # When debugging is on, we process every frame so the video doesn't stutter.  
    FRAME_SKIP = 1 if DEBUG_DISPLAY else 5                
    
    print("Pipeline started. Press 'q' on the video window to stop.")
    
    try:
        red_led.off()
        green_led.off()

        while True:
            raw_frame = picam2.capture_array()
            frame = cv2.resize(raw_frame,(640,480))
            frame_count += 1
            
            if frame_count % FRAME_SKIP != 0:
                continue
                
            results = model(frame, verbose=False)
            person_detected_this_frame = False
            current_ratio = 0.0
            status_text = "GOOD POSTURE"
            status_color = (0, 255, 0)

            for result in results:
                # Only draw the skeleton if we are debugging
                if DEBUG_DISPLAY:
                    annotated_frame = result.plot()
                else:
                    annotated_frame = None

                keypoints = result.keypoints
                
                if keypoints is not None and keypoints.data.shape[0] > 0:
                    person_detected_this_frame = True
                    kp_data = keypoints.data[0].cpu().numpy()
                    posture_res = compute_posture_index(kp_data, tracker_state)
                    
                    if posture_res.posture_ratio is not None:
                        current_ratio = posture_res.posture_ratio
                        
                        if current_ratio < SLOUCH_RATIO_THRESHOLD:
                            green_led.off()
                            status_text = "SLOUCHING!"
                            status_color = (0, 0, 255)
                            
                            if slouch_start_time is None:
                                slouch_start_time = time.time()
                            
                            if (time.time() - slouch_start_time) > SLOUCH_TIME_THRESHOLD:
                                red_led.on()
                                log_event(conn, "SLOUCH_DETECTED", current_ratio)
                                slouch_start_time = time.time() 
                        else:
                            slouch_start_time = None 
                            red_led.off()
                            green_led.on()
            
            if not person_detected_this_frame:
                if DEBUG_DISPLAY:
                    annotated_frame = frame
                status_text = "AWAY"
                status_color = (128, 128, 128)

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
                    tracker_state.clear()
                    
                if not is_present:
                    red_led.off()
                    green_led.off()

            # --- VISUAL DEBUGGING OVERLAY ---
            if DEBUG_DISPLAY and annotated_frame is not None:
                cv2.putText(annotated_frame, f"Ratio: {current_ratio:.2f}", (20, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(annotated_frame, f"Status: {status_text}", (20, 100), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 3)

                cv2.imshow("Pi 5 Posture Test - Press 'q' to quit", annotated_frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    except KeyboardInterrupt:
        print("\nShutting down pipeline safely...")
    except Exception as e:
        print(f"\nCritical Error: {e}")
    finally:
        picam2.stop()
        if DEBUG_DISPLAY:
            cv2.destroyAllWindows()
        conn.close()
        red_led.off()
        green_led.off()

if __name__ == "__main__":
    main()