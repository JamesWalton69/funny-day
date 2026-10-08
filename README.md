# 🎯 6-Player Local AI Reaction Challenge System (`funny-day`)

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10%2B-4285F4.svg)](https://developers.google.com/mediapipe)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.9%2B-5C3EE8.svg)](https://opencv.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An offline, local AI-powered **6-Player "Try Not To Laugh / Don't Flinch" Party Challenge System**.

Running on a single laptop, the system monitors **6 players across 3 cameras** in real time while 40 randomized challenge clips play on a connected TV or second monitor. Using MediaPipe, facial blendshapes, 3D head pose estimation, and iris gaze tracking, the AI detects laughter, smirks, sudden flinches, looking away from the screen, and disappearances, compiling a live leaderboard with instant penalty scoring.

---

## 📺 Dual-Screen Architecture

The application is engineered specifically for a dual-display party setup:

1. **📺 TV Display (`http://127.0.0.1:8000/tv`)**
   - Fullscreen video presentation designed for the living room TV (connected via HDMI).
   - Plays 40 randomized challenge clips with randomized suspense intervals between videos.
   - Synchronized round counter, countdown overlays, and match-over fanfare.

2. **🎛 Host & Referee Dashboard (`http://127.0.0.1:8000/host`)**
   - Displayed on the referee's laptop screen.
   - 3 live MJPEG camera streams with real-time broadcast HUD (bounding boxes, Euler angle gauges, iris vectors).
   - Real-time Leaderboard ranked from 1st (least penalties) to 6th.
   - Live Reaction Event Feed logging every laugh, smirk, head jerk, or look-away with timestamps and intensity scores.
   - Full match controls: **▶ Start Match**, **⏹ Stop All**, **🎯 Calibrate (3s)**, **⏭ Skip**, and **↺ Reset**.
   - Camera source dropdowns and **⇄ Swap** buttons to reassign inputs instantly.

3. **🏆 Match Report & Podium (`http://127.0.0.1:8000/results`)**
   - Post-match podium highlighting the winner and full player penalty breakdown.

---

## 🔌 Hardware Setup (100% Pure USB Cable)

The system is designed to run entirely locally without depending on unstable local Wi-Fi:

| Camera Feed | Physical Hardware | Monitored Players | Stream Endpoint | Connection Type |
| :--- | :--- | :--- | :--- | :--- |
| **Cam 1** | **Logitech 720p Webcam** | **Player 1 & 2** | `Device Index 2` | 🔌 **Direct USB Cable** |
| **Cam 2** | **Phone 1 (Vivo V2437)** | **Player 3 & 4** | `http://127.0.0.1:8081/video` | 🔌 **Pure USB (ADB Forward)** |
| **Cam 3** | **Phone 2 (Redmi 8)** | **Player 5 & 6** | `http://127.0.0.1:8082/video` | 🔌 **Pure USB (ADB Forward)** |

> [!TIP]
> **Virtual Mock Fallback**: If any phone or camera is disconnected, the system automatically falls back to procedural animated video feeds so you can test the game without hardware connected.

---

## 🧠 AI Recognition & Reaction Detection Engine

### 1. Expression-Invariant Rigid Cranial 3D Pose
Standard face tracking often uses chin and mouth corners, causing smiles, laughs, or speaking to falsely register as massive head nods or turns. This system anchors head pose estimation to **8 rigid cranial & nasal landmarks** (nose tip, nose base, nasion, glabella, eye outer corners, and tragus ear anchors), making head tracking **100% immune to facial deformation**.

### 2. Adaptive Dual-Rate EMA Smoothing
Head angles (Yaw, Pitch, Roll) are filtered using an adaptive dual-rate Exponential Moving Average (EMA):
- **Micro-movements & sensor noise ($\Delta < 10^\circ$)**: Heavy smoothing ($\alpha = 0.20$) eliminates jitter and HUD flickering.
- **Deliberate head turns ($\Delta \ge 10^\circ$)**: High responsiveness ($\alpha = 0.85$) tracks fast turns with zero perceptual lag.

### 3. Eye & Iris Gaze Tracking
MediaPipe Iris landmarks (`468` and `473`) are normalized against eye corners and eyelids to compute horizontal ($H_{iris}$) and vertical ($V_{iris}$) gaze ratios. The AI detects when players glance down at their phone or look sideways at other players **even if their head remains pointed at the screen**.

### 4. 52-Blendshape Reaction Detector
Rather than noisy 1-frame derivatives, reactions are analyzed over a **150ms temporal window** matching human neurological reaction onset:
- **Laughter / Smiles**: Evaluates `mouthSmileLeft`, `mouthSmileRight`, and `jawOpen`.
- **Gasps / Surprise**: Evaluates `browInnerUp` and `eyeWideLeft/Right`.
- **Flinches & Recoils**: Detects rapid angular shifts $> 18^\circ$ over 150ms.
- **Posture Shifts**: Detects sudden body/torso displacement (ducking, jumping).
- **Look Aways & Disappearances**: Tracks when players look away or cover their faces.

### 5. Referee Sensitivity Presets
Toggled in 1 click from the dashboard:
- **Casual / Party (Forgiving)**: Allows natural party banter; requires clear laughs/smiles.
- **Normal / Balanced**: Standard tournament challenge thresholds.
- **Strict / Hardcore (Poker Face)**: Detects subtle smirks and eye twitches.

---

## 📊 Penalty Scoring Formula

Players start with **0.0 penalty points**. The player with the lowest score wins:

$$\text{Penalty} = (2.0 \times T_{\text{away}}) + (5.0 \times N_{\text{look\_away}}) + (10.0 \times \sum \text{Intensity}) + (15.0 \times T_{\text{disappear}}) + (1.5 \times (100 - \text{Attention}\%))$$

Where:
- $T_{\text{away}}$ = Total seconds spent looking away from the TV.
- $N_{\text{look\_away}}$ = Number of distinct look-away events.
- $\sum \text{Intensity}$ = Cumulative intensity ($0.0 - 1.0$) of detected reactions.
- $T_{\text{disappear}}$ = Total seconds face disappeared from the camera.
- $\text{Attention}\%$ = Percentage of match time spent facing the TV.

---

## 🚀 Quick Start Guide

### 1. Prerequisites
- **Python 3.10+** (64-bit)
- **Git**
- Two Android phones with **USB Debugging** enabled in Developer Options.
- **IP Webcam** app installed on both phones (available on Google Play Store).

### 2. Installation
```powershell
# Clone the repository
git clone https://github.com/JamesWalton69/funny-day.git
cd funny-day

# Create virtual environment
python -m venv .venv

# Activate virtual environment (Windows PowerShell)
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### 3. MediaPipe Face Landmarker Model
The model is expected at `models/face_landmarker.task`. If missing, download it via PowerShell:
```powershell
Invoke-WebRequest -Uri "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task" -OutFile "models/face_landmarker.task"
```

### 4. Setup Android Phones via USB
1. Plug both Android phones into your laptop using USB cables.
2. Ensure **USB Debugging** is turned on.
3. Open **IP Webcam** on both phones and tap **Start Server** at the bottom.
4. The launcher will automatically forward:
   - Phone 1 $\rightarrow$ `http://127.0.0.1:8081/video`
   - Phone 2 $\rightarrow$ `http://127.0.0.1:8082/video`

### 5. Launch the Challenge
```powershell
python run.py
```
This single command will:
1. Verify challenge videos.
2. Configure pure USB ADB port forwarding for connected phones.
3. Start the FastAPI local server on port `8000`.
4. Open the **Host Dashboard** in your default web browser.

---

## 🎬 Challenge Videos & Downloader

Videos are stored in [`data/videos/`](file:///E:/Coding/Projects/funny-day/data/videos/).

### Option A: Download Curated Sample Clips
```powershell
python scripts/download_sample_videos.py --mode sample
```

### Option B: Download from YouTube via `yt-dlp`
You can download any YouTube video, "Try Not to Laugh" compilation, playlist, or Short directly into the challenge pool:
```powershell
# Download a single YouTube video / Short
python scripts/download_sample_videos.py --yt "https://www.youtube.com/watch?v=VIDEO_ID"

# Download up to 10 videos from a YouTube playlist
python scripts/download_sample_videos.py --yt "https://www.youtube.com/playlist?list=PLAYLIST_ID" --max 10
```

### Option C: Drag & Drop Local Videos
Simply copy any `.mp4`, `.webm`, or `.mkv` files directly into `data/videos/`.

---

## 🧪 Automated Testing

Run the full automated test suite (11 unit tests):
```powershell
.\.venv\Scripts\python.exe -m unittest discover tests
```
Output:
```
Ran 11 tests in 1.591s
OK
```

---

## 📁 Project Structure

```
funny-day/
├── backend/
│   ├── adb_manager.py          # Portable ADB detection & USB port forwarding
│   ├── app.py                  # FastAPI server, MJPEG streaming & WebSockets
│   ├── camera_manager.py       # Multi-threaded grabbers & latency downsampling
│   ├── config.py               # YAML configuration loader & persister
│   ├── game_engine.py          # Match state machine, telemetry & penalty formulas
│   ├── pose_tracker.py         # 3D head pose, adaptive EMA, and iris gaze tracking
│   ├── reaction_detector.py    # 150ms temporal blendshape reaction analyzer
│   └── video_manager.py        # 40-video playlist manager & procedural generator
├── frontend/
│   ├── static/
│   │   ├── css/style.css       # Responsive dark-theme dashboard & TV styles
│   │   └── js/
│   │       ├── dashboard.js    # Live telemetry, leaderboard, HUD & controls
│   │       ├── results.js      # Match results & podium rendering
│   │       └── tv.js           # Fullscreen HDMI TV display sync
│   └── templates/
│       ├── host.html           # Referee / Host monitoring dashboard
│       ├── results.html        # Match report & podium view
│       └── tv.html             # TV screen view
├── models/
│   └── face_landmarker.task    # MediaPipe FaceLandmarker model file
├── scripts/
│   └── download_sample_videos.py # Sample video bundle & yt-dlp downloader
├── tests/
│   ├── test_game_engine.py     # State machine & scoring tests
│   ├── test_pose_tracker.py    # Head pose & filter tests
│   ├── test_reaction_detector.py # Blendshape laughter & flinch tests
│   └── test_video_manager.py   # Playlist tests
├── config.yaml                 # User settings, camera mappings & thresholds
├── requirements.txt            # Python dependencies
├── run.py                      # Single-command application launcher
└── README.md                   # System documentation
```

---

## 📜 License

This project is open-source under the [MIT License](LICENSE).
