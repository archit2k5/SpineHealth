import cv2
import math
import time
import sqlite3
from ultralytics import YOLO
from gpiozero import LED  # <-- New Import

# Initialize the LEDs using their GPIO (BCM) numbers
red_led = LED(17)
green_led = LED(27)

# ... [Keep your init_db and calculate_neck_angle functions exactly the same] ...

def main():
    conn = init_db()
    
    # Load your optimized model (NCNN or ONNX)
    model = YOLO("yolov8n-pose_ncnn_model") 
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    is_present = False
    last_seen_time = time.time()
    slouch_start_time = None
    
    # Thresholds
    SLOUCH_ANGLE_THRESHOLD = 20.0 
    SLOUCH_TIME_THRESHOLD = 5     # Reduced to 5 seconds for immediate LED feedback
    BREAK_TIME_THRESHOLD = 60     
    FRAME_SKIP = 5                
    
    frame_count = 0

    print("Pipeline started. Press Ctrl+C to stop.")
    
    try:
        # Ensure LEDs are off at startup
        red_led.off()
        green_led.off()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            frame_count += 1
            if frame_count % FRAME_SKIP != 0: continue

            results = model(frame, verbose=False)
            person_detected_this_frame = False

            for result in results:
                keypoints = result.keypoints
                
                if keypoints is not None and len(keypoints.xy[0]) > 0:
                    person_detected_this_frame = True
                    kp_array = keypoints.xy[0].cpu().numpy() 
                    conf_array = keypoints.conf[0].cpu().numpy()

                    if conf_array[3] > 0.5 and conf_array[5] > 0.5:
                        l_ear_x, l_ear_y = kp_array[3]
                        l_shoulder_x, l_shoulder_y = kp_array[5]
                        
                        angle = calculate_neck_angle(l_ear_x, l_ear_y, l_shoulder_x, l_shoulder_y)
                        
                        # --- NEW LED LOGIC ---
                        if angle > SLOUCH_ANGLE_THRESHOLD:
                            # Person is crouching/slouching
                            if slouch_start_time is None:
                                slouch_start_time = time.time()
                            
                            # If they slouch longer than the threshold, trigger the Red LED
                            if (time.time() - slouch_start_time) > SLOUCH_TIME_THRESHOLD:
                                red_led.on()
                                green_led.off()
                                # Only log to DB periodically to avoid spam
                                # log_event(conn, "SLOUCH_DETECTED", angle)
                        else:
                            # Good posture
                            slouch_start_time = None 
                            red_led.off()
                            green_led.on()

            # Occupancy State Machine updates
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
                    
                # If no one is at the desk, turn both LEDs off to save power
                if not is_present:
                    red_led.off()
                    green_led.off()

    except KeyboardInterrupt:
        print("Shutting down pipeline...")
    finally:
        cap.release()
        conn.close()
        # Clean up GPIO state on exit
        red_led.off()
        green_led.off()

if __name__ == "__main__":
    main()