# Visual Guide: How the IoT & Hardware System Works

This document translates the technical IoT and Edge AI hardware setup into a simple, highly visual guide and presents the premium dashboard interface used by clinicians to monitor results.

---

## 🎨 1. The Real-Time Hardware & IoT Monitoring Interface

When the physical edge device (Raspberry Pi or Jetson board) detects a tumor and sends the information to the cloud over IoT, the clinician monitors everything on this web dashboard:

```
+-----------------------------------------------------------------------------------------+
| 🌐 ULTRA-XAI: ADVANCED EDGE & IOT MONITORING CONSOLE                                    |
| [ SYSTEM STATUS: CONNECTED ]                      [ EDGE NODE: JETSON-ORIN-01 ]          |
+-----------------------------------------------------------------------------------------+
| 📟 HARDWARE TELEMETRY & PHYSICAL SENSORS                                                |
|                                                                                         |
|  🟢 CPU Temperature: 42°C  |  🔵 Memory Usage: 34%  | 💡 Diagnostic LED: RED ALERT (Active) |
|  🔌 Probe Streaming: USB 3.0 |  📡 IoT Cloud Sync: Synced 2 seconds ago                 |
+-----------------------------------------------------------------------------------------+
| 🩺 INCOMING REAL-TIME SCAN (FROM HANDHELD DEVICE)                                        |
|                                                                                         |
|   [ Live Camera / Probe Input ]      [ Segmented Tumor Mask ]      [ XAI Attention Map ]|
|   +---------------------------+      +----------------------+      +-------------------+|
|   |                           |      |       _______        |      |      _..---.._    ||
|   |        (o) Tumor          | -->  |      /       \       | -->  |    .'  High   '.  ||
|   |         Detected          |      |     |  Mask   |      |      |   /  Influence \  ||
|   |                           |      |      \_______/       |      |    '.   Red   .'  ||
|   +---------------------------+      +----------------------+      +-------------------+|
|                                                                                         |
+-----------------------------------------------------------------------------------------+
| 📊 REAL-TIME IOT DIAGNOSTIC REPORT                                                       |
|                                                                                         |
|  🏥 Patient ID: P-2026-08112              🚨 Alert Status: 🔴 MALIGNANT DETECTED       |
|  ⏱️ Scanning Time: 1.2 seconds            📈 Diagnostic Accuracy: 99.18%              |
|                                                                                         |
|  ⚡ Physical Hardware Triggers Activated:                                               |
|  ✅ GPIO Pin 18: Red Indicator Light ON                                                 |
|  ✅ GPIO Pin 24: Audio Buzzer Sounding Warning                                          |
+-----------------------------------------------------------------------------------------+
| 📥 EMERGENCY REMOTE ACTIONS                                                             |
|  [ Send Alert to On-Duty Radiologist ]    [ Upload Scan directly to Cloud Patient Record ]|
+-----------------------------------------------------------------------------------------+
```

---

## 📋 2. Simplified Operational Guide (How it works in 4 Steps)

Here is exactly how the hardware, the software, and the user interface connect, explained simply:

### Step 1: Scanning the Patient
The doctor places the handheld sensor/camera over the patient's skin. The live video frames are instantly sent over a USB cable to the local **NVIDIA Jetson / Raspberry Pi** board.

### Step 2: The Edge Board Predicts (Local Processing)
The board immediately runs the fast **YOLOv8-Seg** model locally on the incoming frames. It finds the lesion, separates it from background noise, and makes a final **Benign vs. Malignant** decision.

### Step 3: Local Hardware Responds (The Alerts)
If the model determines that the tumor has a high chance of being malignant:
- It sends electricity to a pin to turn on a physical **Bright Red Warning Light**.
- It triggers a physical **Buzzer** to make a warning sound, alerting the clinician in the room immediately.

### Step 4: The Cloud Dashboard Syncs
At the same time, the device uses Wi-Fi to push the diagnostic percentage, the original scan image, and the tumor mask up to the **Cloud database**. The clinician monitoring the patient's records from their laptop instantly sees the visual telemetry update on the **Web Dashboard** shown above.
