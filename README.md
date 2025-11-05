# Dizzy
# 🧠 Drowsiness & Dizziness Detection System

A **real-time computer vision system** that monitors driver alertness by detecting signs of **drowsiness** and **dizziness** using **facial landmark detection** and **head pose estimation**.  
Optimized for vehicle environments with advanced **vibration filtering** to eliminate false alerts caused by road bumps or potholes.

---

## 🚀 Features

### 🔍 Core Detection Capabilities
- **Eye Closure Detection:** Tracks Eye Aspect Ratio (EAR) to detect prolonged eye closure.  
- **Yawn Detection:** Monitors Mouth Aspect Ratio (MAR) for excessive yawning.  
- **Head Pose Monitoring:** Uses head tilt and rotation to detect dizziness or micro-sleep.  
- **Vibration Filtering:** Smooths false positives from vehicle vibrations and uneven roads.

### ⚙️ System Highlights
- Real-time detection with optimized frame rates  
- Non-blocking alert system (thread-safe)  
- Visual feedback overlay with live metrics  
- Configurable detection thresholds  
- Adjustable sensitivity levels  

---

## 🧩 Requirements

### Dependencies
```bash
opencv-python >= 4.5.0
dlib >= 19.22.0
numpy >= 1.19.0
scipy >= 1.5.0
playsound >= 1.2.2   # optional (for audio alerts)

### use python version and dlib verdsion same in this case we have used pyhon 3.12 and same dlib version  12 which is compatible with python.
- landmark file is used in this
shape_predictor_68_face_landmarks.dat
