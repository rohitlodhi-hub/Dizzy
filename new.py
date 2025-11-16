import cv2
import dlib
import numpy as np
from scipy.spatial import distance as dist
from threading import Thread, Lock
from collections import deque
import time
import os
import datetime
import logging
import math
import pygame

# --- Simple logger ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# --- Configuration ---
class Config:
    # Thresholds (fallbacks; may be replaced after calibration)
    EYE_AR_THRESH = 0.25
    MOUTH_AR_THRESH = 0.65

    # Multipliers applied after calibration (threshold = baseline * MULTIPLIER)
    EYE_AR_MULTIPLIER = 0.6  # trigger when EAR drops below 60% of baseline
    MOUTH_AR_MULTIPLIER = 1.6  # trigger when MAR increases beyond 160% of baseline

    # Consecutive frames required to consider a state
    EYE_AR_CONSEC_FRAMES = 15
    YAWN_CONSEC_FRAMES = 10
    HEAD_TILT_CONSEC_FRAMES = 20

    # Head tilt thresholds (in degrees)
    HEAD_TILT_THRESH = 25.0
    HEAD_TILT_SUSTAINED_THRESH = 20.0

    # Head position smoothing & vibration detection
    HEAD_POSITION_BUFFER_SIZE = 10
    VELOCITY_BUFFER_SIZE = 8
    VELOCITY_VARIANCE_THRESHOLD = 4.0  # var of angular velocity beyond which we consider vibration

    # Face detection / tracking
    FACE_DETECTION_INTERVAL = 8  # run full dlib detection every N frames (use tracker otherwise)

    # Alerting
    ALERT_COOLDOWN = 3.0  # seconds between alerts
    ALERT_SAVE_DIR = "alerts"  # where to save screenshots on alerts

    # Camera
    CAMERA_WIDTH = 640
    CAMERA_HEIGHT = 480
    FPS_LIMIT = 30

    # Calibration
    CALIBRATION_FRAMES = 120  # frames to gather baseline EAR/MAR at start

# --- Landmark indices ---
LEFT_EYE = list(range(36, 42))
RIGHT_EYE = list(range(42, 48))
OUTER_MOUTH = list(range(48, 60))

# --- Audio init (pygame mixer) ---
pygame.mixer.init()
try:
    ALERT_SOUND = pygame.mixer.Sound("alert.wav")
except Exception as e:
    logging.warning(f"Could not load alert.wav: {e}")
    ALERT_SOUND = None

# --- Utilities ---
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    if C == 0:
        return 0.0
    return (A + B) / (2.0 * C)

def mouth_aspect_ratio(mouth):
    A = dist.euclidean(mouth[2], mouth[10])
    B = dist.euclidean(mouth[4], mouth[8])
    C = dist.euclidean(mouth[0], mouth[6])
    if C == 0:
        return 0.0
    return (A + B) / (2.0 * C)

def head_pose_estimation(shape, focal_length, center):
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
    # Use RQ decomposition to extract Euler angles
    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
    # angles returned are (roll, pitch, yaw) in degrees in many OpenCV builds;
    # we convert carefully and treat them as pitch, yaw, roll as in original code by mapping
    # Use safe mapping and correct sign if needed via calibration step
    roll = angles[0]
    pitch = angles[1]
    yaw = angles[2]
    return pitch, yaw, roll

# --- Filters ---
class EMAFilter:
    def __init__(self, alpha=0.3):
        self.alpha = alpha
        self.value = None

    def update(self, x):
        if self.value is None:
            self.value = np.array(x, dtype=np.float64)
        else:
            self.value = self.alpha * np.array(x, dtype=np.float64) + (1 - self.alpha) * self.value
        return tuple(self.value.tolist())

class MedianThenEMAFilter:
    """
    Keeps a short buffer, computes median to remove outliers, then applies EMA on medians.
    Used for head pose smoothing.
    """
    def __init__(self, buffer_size=7, ema_alpha=0.35):
        self.buf = deque(maxlen=buffer_size)
        self.ema = EMAFilter(alpha=ema_alpha)

    def update(self, x_tuple):
        self.buf.append(x_tuple)
        arr = np.array(self.buf)
        med = np.median(arr, axis=0)
        return self.ema.update(med)

# --- Alert management ---
class AlertManager:
    def __init__(self):
        self.last_alert_time = 0.0
        self.lock = Lock()

    def trigger_alert(self, alert_type, frame=None):
        now = time.time()
        with self.lock:
            if now - self.last_alert_time < Config.ALERT_COOLDOWN:
                return
            self.last_alert_time = now

        msg = {
            'eye': "⚠️ Drowsiness Detected (Eyes Closed Too Long)",
            'yawn': "⚠️ Drowsiness Detected (Yawning)",
            'head': "⚠️ Dizziness Detected (Unnatural Head Movement)"
        }.get(alert_type, "⚠️ Alert")

        logging.warning(msg)

        # Play sound non-blocking
        if ALERT_SOUND:
            try:
                Thread(target=self._play_sound, daemon=True).start()
            except Exception as e:
                logging.warning(f"Sound thread error: {e}")

        # Save frame snapshot for analysis (non-blocking)
        if frame is not None:
            Thread(target=self._save_frame, args=(frame.copy(), alert_type), daemon=True).start()

    def _play_sound(self):
        try:
            ALERT_SOUND.play()
        except Exception as e:
            logging.warning(f"Alert sound error: {e}")

    def _save_frame(self, frame, alert_type):
        try:
            os.makedirs(Config.ALERT_SAVE_DIR, exist_ok=True)
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            fname = os.path.join(Config.ALERT_SAVE_DIR, f"alert_{alert_type}_{ts}.jpg")
            cv2.imwrite(fname, frame)
            logging.info(f"Saved alert frame to {fname}")
        except Exception as e:
            logging.warning(f"Failed to save alert frame: {e}")

# --- Head position filter with velocity-based vibration detection ---
class HeadPositionFilter:
    def __init__(self, buffer_size=Config.HEAD_POSITION_BUFFER_SIZE, vel_buf_size=Config.VELOCITY_BUFFER_SIZE):
        self.filter = MedianThenEMAFilter(buffer_size=buffer_size, ema_alpha=0.35)
        self.prev = None
        self.vel_buf = deque(maxlen=vel_buf_size)

    def update(self, pitch, yaw, roll):
        # update smoothing filter
        smoothed = self.filter.update((pitch, yaw, roll))
        # compute velocity (difference)
        if self.prev is None:
            vel = (0.0, 0.0, 0.0)
        else:
            vel = (smoothed[0] - self.prev[0], smoothed[1] - self.prev[1], smoothed[2] - self.prev[2])
        self.prev = smoothed
        self.vel_buf.append(vel)
        return smoothed

    def get_velocity_variance(self):
        if len(self.vel_buf) < 3:
            return 0.0, 0.0, 0.0
        arr = np.array(self.vel_buf)
        return float(np.var(arr[:, 0])), float(np.var(arr[:, 1])), float(np.var(arr[:, 2]))

# --- Detection system ---
class DrowsinessDetector:
    def __init__(self, predictor_path):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(predictor_path)
        self.alert_manager = AlertManager()
        self.head_filter = HeadPositionFilter(Config.HEAD_POSITION_BUFFER_SIZE)
        self.counter_eye = 0
        self.counter_yawn = 0
        self.counter_head = 0

        self.focal_length = None
        self.center = None

        # Tracking
        self.tracker = None
        self.tracking = False
        self.detect_frame_count = 0

        # Calibration buffers
        self.calib_ear = []
        self.calib_mar = []
        self.calibrated = False
        self.frame_idx = 0

        # Adaptive thresholds (start with defaults)
        self.eye_thresh = Config.EYE_AR_THRESH
        self.mouth_thresh = Config.MOUTH_AR_THRESH

        # For logging / diagnostics
        self.lock = Lock()

    def initialize_camera_params(self, frame_shape):
        if self.focal_length is None:
            self.focal_length = frame_shape[1]
            self.center = (frame_shape[1] / 2.0, frame_shape[0] / 2.0)

    def _rect_to_bbox(self, rect):
        # dlib rect to (x, y, w, h)
        x = rect.left()
        y = rect.top()
        w = rect.width()
        h = rect.height()
        return (x, y, w, h)

    def _bbox_to_rect(self, bbox):
        x, y, w, h = bbox
        return dlib.rectangle(int(x), int(y), int(x + w), int(y + h))

    def process_frame(self, frame, gray):
        self.initialize_camera_params(frame.shape)
        self.frame_idx += 1
        self.detect_frame_count += 1

        # Decide whether to use tracker
        face_rect = None
        if self.tracking and self.tracker is not None:
            ok, tracked_box = self.tracker.update(frame)
            if ok:
                x, y, w, h = tuple(map(int, tracked_box))
                # convert to dlib rect for predictor
                face_rect = self._bbox_to_rect((x, y, w, h))
            else:
                # tracking failed -> force detection next
                self.tracking = False
                self.tracker = None

        if (not self.tracking) or (self.detect_frame_count >= Config.FACE_DETECTION_INTERVAL):
            faces = self.detector(gray, 0)
            self.detect_frame_count = 0
            if len(faces) > 0:
                # choose largest face (likely the driver)
                face = max(faces, key=lambda r: r.width() * r.height())
                face_rect = face
                # initialize tracker based on face bbox
                x, y, w, h = self._rect_to_bbox(face_rect)
                try:
                    # MOSSE/KCF: MOSSE is fast; if not present use KCF
                    self.tracker = cv2.legacy.TrackerMOSSE_create() if hasattr(cv2.legacy, "TrackerMOSSE_create") else cv2.TrackerMOSSE_create()
                except Exception:
                    try:
                        self.tracker = cv2.TrackerKCF_create()
                    except Exception:
                        self.tracker = None
                if self.tracker is not None:
                    try:
                        self.tracker.init(frame, (x, y, w, h))
                        self.tracking = True
                    except Exception:
                        self.tracker = None
                        self.tracking = False

        # If still no face, return
        if face_rect is None:
            return

        # Extract landmarks
        shape = self.predictor(gray, face_rect)
        shape = np.array([[p.x, p.y] for p in shape.parts()])

        left_eye = shape[LEFT_EYE]
        right_eye = shape[RIGHT_EYE]
        mouth = shape[OUTER_MOUTH]

        ear = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2.0
        mar = mouth_aspect_ratio(mouth)

        # Calibration phase: gather baseline EAR/MAR
        if not self.calibrated:
            self.calib_ear.append(ear)
            self.calib_mar.append(mar)
            if len(self.calib_ear) >= Config.CALIBRATION_FRAMES:
                baseline_ear = float(np.mean(self.calib_ear))
                baseline_mar = float(np.mean(self.calib_mar))
                # Avoid zero or extremely small baselines
                baseline_ear = baseline_ear if baseline_ear > 0.01 else Config.EYE_AR_THRESH
                baseline_mar = baseline_mar if baseline_mar > 0.01 else Config.MOUTH_AR_THRESH

                self.eye_thresh = baseline_ear * Config.EYE_AR_MULTIPLIER
                self.mouth_thresh = baseline_mar * Config.MOUTH_AR_MULTIPLIER
                self.calibrated = True
                logging.info(f"Calibration complete. baseline_ear={baseline_ear:.3f}, baseline_mar={baseline_mar:.3f}")
                logging.info(f"Adaptive thresholds set: eye_thresh={self.eye_thresh:.3f}, mouth_thresh={self.mouth_thresh:.3f}")
        # Head pose
        pitch_raw, yaw_raw, roll_raw = head_pose_estimation(shape, self.focal_length, self.center)
        pitch, yaw, roll = self.head_filter.update(pitch_raw, yaw_raw, roll_raw)
        pitch_var, yaw_var, roll_var = self.head_filter.get_velocity_variance()

        # Update counters
        self._update_counters(ear, mar, pitch, yaw, pitch_var, yaw_var)

        # Check alerts and pass current frame for saving if alert triggers
        self._check_alerts(frame)

        # Draw visualization
        self._draw_visualization(frame, left_eye, right_eye, mouth, ear, mar, pitch, yaw, roll, pitch_var, yaw_var)

    def _update_counters(self, ear, mar, pitch, yaw, pitch_vel_var, yaw_vel_var):
        # Eye and yawn
        eye_thresh = self.eye_thresh if self.calibrated else Config.EYE_AR_THRESH
        mouth_thresh = self.mouth_thresh if self.calibrated else Config.MOUTH_AR_THRESH

        self.counter_eye = self.counter_eye + 1 if ear < eye_thresh else 0
        self.counter_yawn = self.counter_yawn + 1 if mar > mouth_thresh else 0

        # Head tilt with velocity-based vibration detection
        is_high_velocity = (pitch_vel_var > Config.VELOCITY_VARIANCE_THRESHOLD or
                            yaw_vel_var > Config.VELOCITY_VARIANCE_THRESHOLD)

        if is_high_velocity:
            # treat as vibration; reset head counter
            self.counter_head = 0
        else:
            sustained_tilt = (abs(pitch) > Config.HEAD_TILT_SUSTAINED_THRESH or abs(yaw) > Config.HEAD_TILT_SUSTAINED_THRESH)
            self.counter_head = self.counter_head + 1 if sustained_tilt else 0

    def _check_alerts(self, frame):
        if self.counter_eye >= Config.EYE_AR_CONSEC_FRAMES:
            self.alert_manager.trigger_alert('eye', frame)
            self.counter_eye = 0
        elif self.counter_yawn >= Config.YAWN_CONSEC_FRAMES:
            self.alert_manager.trigger_alert('yawn', frame)
            self.counter_yawn = 0
        elif self.counter_head >= Config.HEAD_TILT_CONSEC_FRAMES:
            self.alert_manager.trigger_alert('head', frame)
            self.counter_head = 0

    def _draw_visualization(self, frame, left_eye, right_eye, mouth, ear, mar, pitch, yaw, roll, pvar, yvar):
        for (x, y) in np.concatenate((left_eye, right_eye, mouth)):
            cv2.circle(frame, (int(x), int(y)), 2, (0, 255, 0), -1)

        metrics = [
            (f"EAR: {ear:.2f}", (10, 20)),
            (f"MAR: {mar:.2f}", (10, 45)),
            (f"Head: P={pitch:.1f} Y={yaw:.1f} R={roll:.1f}", (10, 70)),
            (f"VelVar: P={pvar:.2f} Y={yvar:.2f}", (10, 95)),
        ]

        if not self.calibrated:
            metrics.append((f"Calibrating... ({len(self.calib_ear)}/{Config.CALIBRATION_FRAMES})", (10, 125)))
        else:
            metrics.append((f"EyeThr: {self.eye_thresh:.3f} MouthThr: {self.mouth_thresh:.3f}", (10, 125)))

        for text, pos in metrics:
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (pos[0] - 3, pos[1] - th - 3), (pos[0] + tw + 3, pos[1] + 3), (0, 0, 0), -1)
            cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

# --- Main ---
def main():
    predictor_path = r"C:\Users\rohit\Downloads\PRogram\Dizzy\shape_predictor_68_face_landmarks.dat"
    if not os.path.exists(predictor_path):
        logging.error(f"Predictor not found at {predictor_path}. Download or update the path.")
        return

    detector = DrowsinessDetector(predictor_path)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, Config.CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, Config.FPS_LIMIT)

    if not cap.isOpened():
        logging.error("Could not open camera")
        return

    logging.info("Drowsiness Detection System Active (with tracker, smoothing, and calibration)")
    logging.info("Press 'q' to quit")

    frame_time = 1.0 / Config.FPS_LIMIT

    try:
        while True:
            start_time = time.time()
            ret, frame = cap.read()
            if not ret:
                logging.error("Failed to capture frame")
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
        logging.info("Shutting down (KeyboardInterrupt)")
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
