import cv2
import dlib
import numpy as np
from scipy.spatial import distance as dist
from threading import Thread
from collections import deque
import time

# --- Configuration ---
class Config:
    EYE_AR_THRESH = 0.25
    EYE_AR_CONSEC_FRAMES = 15
    MOUTH_AR_THRESH = 0.65
    HEAD_TILT_THRESH = 25.0  # Increased threshold for head tilt
    HEAD_TILT_SUSTAINED_THRESH = 20.0  # Threshold for sustained tilt
    YAWN_CONSEC_FRAMES = 10
    HEAD_TILT_CONSEC_FRAMES = 20  # Increased frames needed for alert
    HEAD_POSITION_BUFFER_SIZE = 10  # Frames to average for smoothing
    ALERT_COOLDOWN = 3.0  # seconds between alerts
    CAMERA_WIDTH = 640
    CAMERA_HEIGHT = 480
    FPS_LIMIT = 30

# --- Landmark indices ---
LEFT_EYE = list(range(36, 42))
RIGHT_EYE = list(range(42, 48))
OUTER_MOUTH = list(range(48, 60))

# --- Optimized Alert System ---
class AlertManager:
    def __init__(self):
        self.last_alert_time = 0
        self.alert_messages = {
            'eye': "⚠️ Drowsiness Detected (Eyes Closed Too Long)",
            'yawn': "⚠️ Drowsiness Detected (Yawning)",
            'head': "⚠️ Dizziness Detected (Unnatural Head Movement)"
        }
    
    def trigger_alert(self, alert_type):
        current_time = time.time()
        if current_time - self.last_alert_time >= Config.ALERT_COOLDOWN:
            print(self.alert_messages.get(alert_type, "⚠️ Alert"))
            self.last_alert_time = current_time
            # Non-blocking alert - using thread to avoid blocking main loop
            Thread(target=self._play_sound, daemon=True).start()
    
    def _play_sound(self):
        try:
            from playsound import playsound
            playsound("alert.wav", block=False)
        except Exception as e:
            print(f"Alert sound error: {e}")

# --- Head Position Smoothing ---
class HeadPositionFilter:
    """Filters head position to ignore vibrations and sudden movements"""
    def __init__(self, buffer_size=10):
        self.pitch_buffer = deque(maxlen=buffer_size)
        self.yaw_buffer = deque(maxlen=buffer_size)
        self.roll_buffer = deque(maxlen=buffer_size)
    
    def update(self, pitch, yaw, roll):
        """Add new values and return smoothed averages"""
        self.pitch_buffer.append(pitch)
        self.yaw_buffer.append(yaw)
        self.roll_buffer.append(roll)
        
        return (
            np.mean(self.pitch_buffer),
            np.mean(self.yaw_buffer),
            np.mean(self.roll_buffer)
        )
    
    def get_variance(self):
        """Get variance to detect sudden movements"""
        if len(self.pitch_buffer) < 3:
            return 0.0, 0.0, 0.0
        
        return (
            np.var(self.pitch_buffer),
            np.var(self.yaw_buffer),
            np.var(self.roll_buffer)
        )

# --- Utility Functions ---
def eye_aspect_ratio(eye):
    """Calculate Eye Aspect Ratio (EAR)"""
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

def mouth_aspect_ratio(mouth):
    """Calculate Mouth Aspect Ratio (MAR)"""
    A = dist.euclidean(mouth[2], mouth[10])
    B = dist.euclidean(mouth[4], mouth[8])
    C = dist.euclidean(mouth[0], mouth[6])
    return (A + B) / (2.0 * C)

def head_pose_estimation(shape, focal_length, center):
    """Estimate head pose angles with cached camera parameters"""
    image_points = np.array([
        shape[30], shape[8], shape[36],
        shape[45], shape[48], shape[54]
    ], dtype=np.float64)

    model_points = np.array([
        (0.0, 0.0, 0.0),
        (0.0, -330.0, -65.0),
        (-225.0, 170.0, -135.0),
        (225.0, 170.0, -135.0),
        (-150.0, -150.0, -125.0),
        (150.0, -150.0, -125.0)
    ], dtype=np.float64)

    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype=np.float64)

    dist_coeffs = np.zeros((4, 1))
    
    success, rotation_vector, _ = cv2.solvePnP(
        model_points, image_points, camera_matrix, dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )
    
    if not success:
        return 0.0, 0.0, 0.0
    
    rmat, _ = cv2.Rodrigues(rotation_vector)
    angles, *_ = cv2.RQDecomp3x3(rmat)
    pitch, yaw, roll = np.array(angles) * 180.0
    
    return pitch, yaw, roll

# --- Detection System ---
class DrowsinessDetector:
    def __init__(self, predictor_path):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(predictor_path)
        self.alert_manager = AlertManager()
        self.head_filter = HeadPositionFilter(Config.HEAD_POSITION_BUFFER_SIZE)
        
        # Counters
        self.counter_eye = 0
        self.counter_yawn = 0
        self.counter_head = 0
        
        # Cache camera parameters
        self.focal_length = None
        self.center = None
        
        # Track if high variance (vibration) is detected
        self.high_variance_frames = 0
    
    def initialize_camera_params(self, frame_shape):
        """Initialize and cache camera parameters"""
        if self.focal_length is None:
            self.focal_length = frame_shape[1]
            self.center = (frame_shape[1] / 2, frame_shape[0] / 2)
    
    def process_frame(self, frame, gray):
        """Process a single frame for drowsiness detection"""
        self.initialize_camera_params(frame.shape)
        faces = self.detector(gray, 0)  # 0 = no upsampling for speed
        
        for face in faces:
            shape = self.predictor(gray, face)
            shape = np.array([[p.x, p.y] for p in shape.parts()])
            
            # Extract features
            left_eye = shape[LEFT_EYE]
            right_eye = shape[RIGHT_EYE]
            mouth = shape[OUTER_MOUTH]
            
            # Calculate ratios
            ear = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2.0
            mar = mouth_aspect_ratio(mouth)
            
            # Get raw head pose
            pitch_raw, yaw_raw, roll_raw = head_pose_estimation(shape, self.focal_length, self.center)
            
            # Apply smoothing filter to head position
            pitch, yaw, roll = self.head_filter.update(pitch_raw, yaw_raw, roll_raw)
            
            # Get variance to detect vibrations
            pitch_var, yaw_var, roll_var = self.head_filter.get_variance()
            
            # Update counters and check for alerts
            self._update_counters(ear, mar, pitch, yaw, pitch_var, yaw_var)
            self._check_alerts()
            
            # Visualize
            self._draw_visualization(frame, left_eye, right_eye, mouth, ear, mar, 
                                    pitch, yaw, roll, pitch_var, yaw_var)
    
    def _update_counters(self, ear, mar, pitch, yaw, pitch_var, yaw_var):
        """Update detection counters with vibration filtering"""
        # Eye and yawn counters (unchanged)
        self.counter_eye = self.counter_eye + 1 if ear < Config.EYE_AR_THRESH else 0
        self.counter_yawn = self.counter_yawn + 1 if mar > Config.MOUTH_AR_THRESH else 0
        
        # Head tilt with vibration detection
        # High variance indicates sudden movement (vibration/pothole)
        variance_threshold = 50.0  # Threshold for detecting vibrations
        is_high_variance = (pitch_var > variance_threshold or yaw_var > variance_threshold)
        
        if is_high_variance:
            self.high_variance_frames += 1
            # Reset counter during high variance periods (vibrations)
            self.counter_head = 0
        else:
            self.high_variance_frames = max(0, self.high_variance_frames - 1)
            
            # Only count sustained tilts when not experiencing vibrations
            if self.high_variance_frames == 0:
                # Use smoothed values and higher threshold
                sustained_tilt = (abs(pitch) > Config.HEAD_TILT_SUSTAINED_THRESH or 
                                 abs(yaw) > Config.HEAD_TILT_SUSTAINED_THRESH)
                self.counter_head = self.counter_head + 1 if sustained_tilt else 0
            else:
                self.counter_head = 0
    
    def _check_alerts(self):
        """Check counters and trigger alerts"""
        if self.counter_eye >= Config.EYE_AR_CONSEC_FRAMES:
            self.alert_manager.trigger_alert('eye')
            self.counter_eye = 0
        elif self.counter_yawn >= Config.YAWN_CONSEC_FRAMES:
            self.alert_manager.trigger_alert('yawn')
            self.counter_yawn = 0
        elif self.counter_head >= Config.HEAD_TILT_CONSEC_FRAMES:
            self.alert_manager.trigger_alert('head')
            self.counter_head = 0
    
    def _draw_visualization(self, frame, left_eye, right_eye, mouth, ear, mar, 
                           pitch, yaw, roll, pitch_var, yaw_var):
        """Draw landmarks and metrics on frame"""
        # Draw landmarks
        for (x, y) in np.concatenate((left_eye, right_eye, mouth)):
            cv2.circle(frame, (int(x), int(y)), 2, (0, 255, 0), -1)
        
        # Draw metrics with background for better visibility
        metrics = [
            (f"EAR: {ear:.2f}", (30, 30), (0, 255, 255)),
            (f"MAR: {mar:.2f}", (30, 60), (255, 255, 0)),
            (f"Head: P={pitch:.1f} Y={yaw:.1f} R={roll:.1f}", (30, 90), (255, 0, 255)),
            (f"Variance: P={pitch_var:.1f} Y={yaw_var:.1f}", (30, 120), (200, 200, 200))
        ]
        
        # Add vibration indicator
        if self.high_variance_frames > 0:
            metrics.append((f"Vibration Detected - Filtering", (30, 150), (255, 165, 0)))
        
        for text, pos, color in metrics:
            # Draw background rectangle
            (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (pos[0] - 5, pos[1] - text_height - 5),
                         (pos[0] + text_width + 5, pos[1] + 5), (0, 0, 0), -1)
            cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

# --- Main Function ---
def main():
    # Initialize detector
    predictor_path = r"C:\Users\rohit\Downloads\PRogram\Dizzy\shape_predictor_68_face_landmarks.dat"
    detector = DrowsinessDetector(predictor_path)
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, Config.CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, Config.FPS_LIMIT)
    
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    print("Drowsiness Detection System Active")
    print("Press 'q' to quit")
    print("System now filters vibrations and pothole movements")
    
    frame_time = 1.0 / Config.FPS_LIMIT
    
    try:
        while True:
            start_time = time.time()
            
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture frame")
                break
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            detector.process_frame(frame, gray)
            
            cv2.imshow("Drowsiness & Dizziness Detection", frame)
            
            # Frame rate limiting
            elapsed = time.time() - start_time
            if elapsed < frame_time:
                time.sleep(frame_time - elapsed)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()