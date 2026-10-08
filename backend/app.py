from __future__ import annotations
import asyncio
import json
import threading
import time
import sys
from pathlib import Path
from typing import Dict, Any, List

# Safe UTF-8 encoding for Windows console
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import cv2
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from backend.config import load_config, save_config
from backend.adb_manager import ADBManager
from backend.video_manager import VideoManager
from backend.pose_tracker import PoseTracker, PlayerPoseData
from backend.reaction_detector import ReactionDetector, ReactionEvent
from backend.game_engine import GameEngine, MatchState
from backend.camera_manager import CameraManager

PROJECT_ROOT = Path(__file__).resolve().parent.parent
STATIC_DIR = PROJECT_ROOT / "frontend" / "static"
TEMPLATES_DIR = PROJECT_ROOT / "frontend" / "templates"
VIDEOS_DIR = PROJECT_ROOT / "data" / "videos"
MATCHES_DIR = PROJECT_ROOT / "data" / "matches"

app = FastAPI(title="6-Player AI Reaction Challenge")

# Mount Static & Video assets
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
app.mount("/videos", StaticFiles(directory=str(VIDEOS_DIR)), name="videos")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# Core System Instances
config = load_config()
adb_mgr = ADBManager()
video_mgr = VideoManager(video_dir=VIDEOS_DIR, target_count=config.get("match", {}).get("total_videos", 40))
pose_tracker = PoseTracker(gaze_thresholds=config.get("gaze_thresholds"))
reaction_detector = ReactionDetector(thresholds=config.get("reaction_thresholds"))
game_engine = GameEngine(config=config, video_manager=video_mgr)
camera_mgr = CameraManager(config=config)

# WebSocket connection manager
active_websockets: List[WebSocket] = []
ws_lock = threading.Lock()

# Background AI Loop
is_ai_running = False
ai_thread: threading.Thread = None


def ai_processing_loop():
    """Real-time multi-camera tracking & reaction detection thread."""
    global is_ai_running
    camera_mgr.start_all()

    player_names = config.get("players", {})

    cam_player_mapping = {
        1: (1, 2),  # Cam 1 -> Players 1 & 2
        2: (3, 4),  # Cam 2 -> Players 3 & 4
        3: (5, 6)   # Cam 3 -> Players 5 & 6
    }

    loop_interval = 0.033  # ~30 FPS

    while is_ai_running:
        loop_start = time.time()
        current_poses: Dict[int, PlayerPoseData] = {}
        all_new_events: List[ReactionEvent] = []

        # Process each camera feed
        for cam_id, (p_left, p_right) in cam_player_mapping.items():
            frame = camera_mgr.get_raw_frame(cam_id)
            if frame is None:
                continue

            # Run 3D head pose and gaze estimation
            try:
                left_pose, right_pose = pose_tracker.process_camera_frame(frame, p_left, p_right)
                current_poses[p_left] = left_pose
                current_poses[p_right] = right_pose

                # Check reactions for both players
                ev_left = reaction_detector.check_player_reactions(
                    left_pose,
                    game_time_sec=game_engine.active_play_time,
                    current_video_round=game_engine.current_round
                )
                ev_right = reaction_detector.check_player_reactions(
                    right_pose,
                    game_time_sec=game_engine.active_play_time,
                    current_video_round=game_engine.current_round
                )
                all_new_events.extend(ev_left)
                all_new_events.extend(ev_right)

                # Render HUD onto frame and store
                annotated = pose_tracker.draw_hud(frame, left_pose, right_pose, player_names)
                camera_mgr.store_annotated_frame(cam_id, annotated)
            except Exception as e:
                # Fallback on frame error
                camera_mgr.store_annotated_frame(cam_id, frame)

        # Game Engine Tick
        tick_result = game_engine.tick(current_poses, all_new_events)

        # Check if calibration completed
        if tick_result.get("calibration_finished"):
            baselines = tick_result.get("baselines", {})
            for pid, (by, bp, br) in baselines.items():
                pose_tracker.set_calibration(pid, by, bp, br)
            print("Gaze calibration complete! Updated baselines:", baselines)

        # Prepare Real-time Telemetry Payload
        telemetry = {
            "type": "telemetry",
            "timestamp": time.time(),
            "game_state": game_engine.get_game_state_payload(),
            "player_poses": {
                pid: {
                    "player_id": pid,
                    "face_detected": pose.face_detected,
                    "yaw": pose.calibrated_yaw,
                    "pitch": pose.calibrated_pitch,
                    "roll": pose.calibrated_roll,
                    "gaze_state": pose.gaze_state.value,
                    "mar": pose.mouth_aspect_ratio,
                    "state_duration": pose.time_in_current_state
                }
                for pid, pose in current_poses.items()
            },
            "new_events": [e.to_dict() for e in all_new_events]
        }

        # Broadcast via WebSockets
        broadcast_telemetry(telemetry)

        # Sleep to maintain ~30 FPS
        elapsed = time.time() - loop_start
        sleep_time = max(0.005, loop_interval - elapsed)
        time.sleep(sleep_time)


def broadcast_telemetry(data: Dict[str, Any]):
    msg_str = json.dumps(data)
    with ws_lock:
        stale = []
        for ws in active_websockets:
            try:
                asyncio.run(ws.send_text(msg_str))
            except Exception:
                stale.append(ws)
        for s in stale:
            if s in active_websockets:
                active_websockets.remove(s)


@app.on_event("startup")
def startup_event():
    global is_ai_running, ai_thread
    # Ensure videos are available
    video_mgr.refresh_and_ensure_videos()
    # Start AI processing thread
    is_ai_running = True
    ai_thread = threading.Thread(target=ai_processing_loop, daemon=True)
    ai_thread.start()
    print("AI Processing & Camera Pipeline started.")


@app.on_event("shutdown")
def shutdown_event():
    global is_ai_running
    is_ai_running = False
    camera_mgr.stop_all()


# -------------------------------------------------------------
# Frontend Page Routes
# -------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
@app.get("/host", response_class=HTMLResponse)
def get_host_dashboard(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="host.html",
        context={
            "config": config,
            "players": config.get("players", {})
        }
    )


@app.get("/tv", response_class=HTMLResponse)
def get_tv_screen(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="tv.html",
        context={
            "config": config
        }
    )


@app.get("/results", response_class=HTMLResponse)
def get_results_screen(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="results.html",
        context={
            "config": config
        }
    )


# -------------------------------------------------------------
# Video Streaming Route (MJPEG)
# -------------------------------------------------------------
@app.get("/api/stream/cam/{cam_id}")
def stream_camera(cam_id: int):
    return StreamingResponse(
        camera_mgr.generate_mjpeg_stream(cam_id),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.get("/api/cameras/available")
def get_available_cameras():
    return camera_mgr.scan_available_sources()


@app.post("/api/cameras/assign")
async def assign_camera(request: Request):
    global config
    data = await request.json()
    cam_id = int(data.get("cam_id", 1))
    source = data.get("source")
    name = data.get("name")

    camera_mgr.update_camera_source(cam_id, source)

    cam_key = f"cam{cam_id}"
    if "cameras" in config and cam_key in config["cameras"]:
        config["cameras"][cam_key]["source"] = source
        if name:
            config["cameras"][cam_key]["name"] = name
        save_config(config)

    return {"success": True, "cam_id": cam_id, "source": source}


# -------------------------------------------------------------
# Game API Control Routes
# -------------------------------------------------------------
@app.post("/api/start")
def start_match():
    game_engine.start_match()
    return {"success": True, "message": "Match started"}


@app.post("/api/stop")
def stop_match():
    game_engine.stop_match()
    return {"success": True, "message": "Match stopped"}


@app.post("/api/calibrate")
def trigger_calibration():
    game_engine.start_calibration()
    return {"success": True, "message": "Calibration started (3s)"}


@app.post("/api/skip_video")
def skip_video():
    game_engine.on_video_finished()
    return {"success": True, "message": "Skipped to next video"}


@app.post("/api/video_done")
def video_done():
    game_engine.on_video_finished()
    return {"success": True, "message": "Round completed"}


@app.post("/api/sensitivity")
async def update_sensitivity(request: Request):
    data = await request.json()
    preset = data.get("preset", "balanced")
    if preset == "party":
        pose_tracker.gaze_thresholds["yaw_sideways_deg"] = 28.0
        pose_tracker.gaze_thresholds["pitch_down_deg"] = -22.0
        reaction_detector.thresholds["smile_absolute_thresh"] = 0.50
        reaction_detector.thresholds["smile_onset_thresh"] = 0.30
        reaction_detector.thresholds["head_flinch_angle_deg"] = 24.0
    elif preset == "strict":
        pose_tracker.gaze_thresholds["yaw_sideways_deg"] = 18.0
        pose_tracker.gaze_thresholds["pitch_down_deg"] = -14.0
        reaction_detector.thresholds["smile_absolute_thresh"] = 0.32
        reaction_detector.thresholds["smile_onset_thresh"] = 0.16
        reaction_detector.thresholds["head_flinch_angle_deg"] = 14.0
    else:  # balanced
        pose_tracker.gaze_thresholds["yaw_sideways_deg"] = 24.0
        pose_tracker.gaze_thresholds["pitch_down_deg"] = -18.0
        reaction_detector.thresholds["smile_absolute_thresh"] = 0.40
        reaction_detector.thresholds["smile_onset_thresh"] = 0.22
        reaction_detector.thresholds["head_flinch_angle_deg"] = 18.0
    return {"success": True, "preset": preset}


@app.post("/api/reset")
def reset_match():
    game_engine.reset_player_stats()
    game_engine.transition_to(MatchState.LOBBY)
    return {"success": True, "message": "Match reset to lobby"}


@app.get("/api/state")
def get_state():
    return game_engine.get_game_state_payload()


# -------------------------------------------------------------
# ADB & Phone Routes
# -------------------------------------------------------------
@app.get("/api/adb/status")
def get_adb_status():
    p1 = config.get("adb", {}).get("phone1_port", 8081)
    p2 = config.get("adb", {}).get("phone2_port", 8082)
    return adb_mgr.get_status(p1, p2)


@app.post("/api/adb/setup")
def setup_adb():
    p1 = config.get("adb", {}).get("phone1_port", 8081)
    p2 = config.get("adb", {}).get("phone2_port", 8082)
    res = adb_mgr.setup_port_forwarding(p1, p2)
    return res


# -------------------------------------------------------------
# Config & Export Routes
# -------------------------------------------------------------
@app.get("/api/config")
def get_config_endpoint():
    return config


@app.post("/api/config")
async def update_config_endpoint(request: Request):
    global config
    new_data = await request.json()
    config.update(new_data)
    save_config(config)
    return {"success": True, "config": config}


@app.get("/api/export/latest")
def export_latest():
    files = sorted(MATCHES_DIR.glob("match_*.json"))
    if files:
        return FileResponse(str(files[-1]), filename=files[-1].name, media_type="application/json")
    return JSONResponse(status_code=404, content={"error": "No match logs found"})


# -------------------------------------------------------------
# WebSocket Telemetry Channel
# -------------------------------------------------------------
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    with ws_lock:
        active_websockets.append(websocket)
    try:
        while True:
            # Keep socket alive and receive client commands
            data = await websocket.receive_text()
            cmd = json.loads(data)
            action = cmd.get("action")
            if action == "start":
                game_engine.start_match()
            elif action == "stop":
                game_engine.stop_match()
            elif action == "calibrate":
                game_engine.start_calibration()
            elif action == "skip":
                game_engine.on_video_finished()
            elif action == "reset":
                game_engine.reset_player_stats()
                game_engine.transition_to(MatchState.LOBBY)
    except WebSocketDisconnect:
        with ws_lock:
            if websocket in active_websockets:
                active_websockets.remove(websocket)
    except Exception:
        with ws_lock:
            if websocket in active_websockets:
                active_websockets.remove(websocket)
