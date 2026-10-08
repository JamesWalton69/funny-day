from __future__ import annotations
import math
import threading
import time
from typing import Dict, Any, Optional, Tuple, Generator
import cv2
import numpy as np


class CameraWorker:
    def __init__(self, cam_id: int, name: str, source: Any, flip_horizontal: bool = False, mock_enabled: bool = True):
        self.cam_id = cam_id
        self.name = name
        self.source = source
        self.flip_horizontal = flip_horizontal
        self.mock_enabled = mock_enabled

        self.cap: Optional[cv2.VideoCapture] = None
        self.is_running = False
        self.thread: Optional[threading.Thread] = None

        self.lock = threading.Lock()
        self.latest_frame: Optional[np.ndarray] = None
        self.last_frame_time: float = 0.0
        self.is_connected = False

        # Mock frame animation state
        self._mock_start_time = time.time()

    def start(self):
        self.is_running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.is_running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        self._release_cap()

    def _release_cap(self):
        if self.cap:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None
        self.is_connected = False

    def update_source(self, new_source: Any):
        with self.lock:
            self.source = new_source
            self._release_cap()

    def _open_capture(self) -> bool:
        self._release_cap()
        if str(self.source).lower() == "mock":
            self.is_connected = False
            return False
        try:
            # If source is an integer or string of int
            if isinstance(self.source, int) or (isinstance(self.source, str) and self.source.isdigit()):
                idx = int(self.source)
                # On Windows, try DirectShow for USB cameras
                self.cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
                if not self.cap.isOpened():
                    self.cap = cv2.VideoCapture(idx)
            else:
                # URL stream (e.g. http://127.0.0.1:8081/video)
                self.cap = cv2.VideoCapture(str(self.source))

            if self.cap and self.cap.isOpened():
                # Optimize resolution and buffer size
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                self.is_connected = True
                return True
        except Exception as e:
            print(f"[Cam {self.cam_id}] Open failed for {self.source}: {e}")

        self.is_connected = False
        return False

    def _capture_loop(self):
        retry_delay = 2.0
        last_retry = 0.0

        while self.is_running:
            now = time.time()

            if not self.is_connected:
                if now - last_retry >= retry_delay:
                    last_retry = now
                    self._open_capture()

            if self.is_connected and self.cap:
                ret, frame = self.cap.read()
                if ret and frame is not None:
                    # Immediately downsample high-resolution feeds (e.g. 1080p phone cameras) to 640px
                    h, w = frame.shape[:2]
                    if w > 640:
                        scale = 640.0 / w
                        frame = cv2.resize(frame, (640, int(h * scale)), interpolation=cv2.INTER_LINEAR)

                    if self.flip_horizontal:
                        frame = cv2.flip(frame, 1)

                    with self.lock:
                        self.latest_frame = frame
                        self.last_frame_time = now
                else:
                    # Connection dropped
                    self.is_connected = False
                    self._release_cap()
            else:
                # Produce synthetic mock frame if enabled
                if self.mock_enabled:
                    frame = self._generate_mock_frame(now)
                    with self.lock:
                        self.latest_frame = frame
                        self.last_frame_time = now

                time.sleep(0.033)  # ~30 FPS

    def get_latest_frame(self) -> Optional[np.ndarray]:
        with self.lock:
            if self.latest_frame is not None:
                return self.latest_frame.copy()
            return None

    def _generate_mock_frame(self, now: float) -> np.ndarray:
        """Generates a synthetic 2-player video feed for offline testing."""
        w, h = 640, 480
        frame = np.full((h, w, 3), (25, 25, 30), dtype=np.uint8)

        t = now - self._mock_start_time

        # Background room ambience lines
        cv2.line(frame, (0, 360), (w, 360), (45, 45, 50), 2)
        cv2.line(frame, (w // 2, 0), (w // 2, h), (40, 40, 45), 1)

        # Cam Header Tag
        cv2.putText(frame, f"[VIRTUAL MOCK CAM {self.cam_id}] {self.name}", (15, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 180, 255), 1, cv2.LINE_AA)

        # Player 1 (Left side, x ~ 160)
        p1_x = 160 + int(8 * math.sin(t * 1.5))
        p1_y = 230 + int(4 * math.cos(t * 1.2))
        self._draw_mock_person(frame, p1_x, p1_y, (60, 140, 240), t, is_left=True)

        # Player 2 (Right side, x ~ 480)
        p2_x = 480 + int(8 * math.sin(t * 1.8 + 1.0))
        p2_y = 230 + int(4 * math.cos(t * 1.4 + 1.0))
        self._draw_mock_person(frame, p2_x, p2_y, (240, 140, 60), t, is_left=False)

        return frame

    def _draw_mock_person(self, img: np.ndarray, cx: int, cy: int, shirt_color: tuple, t: float, is_left: bool):
        # Torso
        cv2.ellipse(img, (cx, cy + 130), (75, 90), 0, 0, 360, shirt_color, -1)
        # Head
        cv2.circle(img, (cx, cy), 50, (190, 210, 235), -1)

        # Natural looking / gaze behavior with occasional subtle looking away
        offset_yaw = 12 * math.sin(t * 0.8 if is_left else t * 0.9 + 2.0)
        eye_y = cy - 8
        left_eye_x = cx - 18 + int(offset_yaw * 0.3)
        right_eye_x = cx + 18 + int(offset_yaw * 0.3)

        # Eyes
        cv2.circle(img, (left_eye_x, eye_y), 5, (50, 50, 50), -1)
        cv2.circle(img, (right_eye_x, eye_y), 5, (50, 50, 50), -1)

        # Nose tip
        nose_x = cx + int(offset_yaw * 0.5)
        nose_y = cy + 10
        cv2.circle(img, (nose_x, nose_y), 3, (120, 140, 180), -1)

        # Mouth (occasional smile/jaw drop)
        smile_delta = int(4 * math.sin(t * 2.5))
        cv2.ellipse(img, (cx + int(offset_yaw * 0.3), cy + 28), (14, max(2, 4 + smile_delta)), 0, 0, 180, (80, 80, 140), 2)


class CameraManager:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.workers: Dict[int, CameraWorker] = {}
        self.annotated_frames: Dict[int, np.ndarray] = {}
        self.frame_locks: Dict[int, threading.Lock] = {i: threading.Lock() for i in [1, 2, 3]}

        mock_enabled = config.get("mock_fallback_enabled", True)
        cameras_cfg = config.get("cameras", {})

        for key, ccfg in cameras_cfg.items():
            cid = ccfg.get("id", 1)
            name = ccfg.get("name", f"Cam {cid}")
            source = ccfg.get("source", 0)
            flip = ccfg.get("flip_horizontal", False)

            worker = CameraWorker(
                cam_id=cid,
                name=name,
                source=source,
                flip_horizontal=flip,
                mock_enabled=mock_enabled
            )
            self.workers[cid] = worker

    def start_all(self):
        for worker in self.workers.values():
            worker.start()

    def stop_all(self):
        for worker in self.workers.values():
            worker.stop()

    def get_raw_frame(self, cam_id: int) -> Optional[np.ndarray]:
        worker = self.workers.get(cam_id)
        if worker:
            return worker.get_latest_frame()
        return None

    def store_annotated_frame(self, cam_id: int, frame: np.ndarray):
        with self.frame_locks[cam_id]:
            self.annotated_frames[cam_id] = frame.copy()

    def get_annotated_frame(self, cam_id: int) -> Optional[np.ndarray]:
        with self.frame_locks[cam_id]:
            if cam_id in self.annotated_frames:
                return self.annotated_frames[cam_id].copy()
            return self.get_raw_frame(cam_id)

    def update_camera_source(self, cam_id: int, new_source: Any):
        worker = self.workers.get(cam_id)
        if worker:
            if isinstance(new_source, str) and new_source.isdigit():
                new_source = int(new_source)
            worker.update_source(new_source)

    def scan_available_sources(self) -> list:
        sources = []
        # 1. Probing USB cameras (0 to 5)
        for i in range(5):
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
            if cap.isOpened():
                ret, _ = cap.read()
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                sources.append({
                    "id": str(i),
                    "name": f"USB Camera Index {i} ({w}x{h})",
                    "type": "usb",
                    "active": True
                })
                cap.release()

        # 2. Pure USB Cable streams via ADB port forwarding (Zero Wi-Fi)
        usb_phones = [
            ("http://127.0.0.1:8081/video", "Phone 1 [USB Cable via ADB Port 8081]"),
            ("http://127.0.0.1:8082/video", "Phone 2 [USB Cable via ADB Port 8082]")
        ]
        import urllib.request
        for url, label in usb_phones:
            sources.append({
                "id": url,
                "name": label,
                "type": "usb_phone",
                "active": True
            })

        # 3. Wi-Fi IP streams (Optional fallback)
        wifi_phones = [
            ("http://10.248.73.38:8080/video", "Phone 1: Vivo V2437 [Wi-Fi]"),
            ("http://10.248.73.237:8080/video", "Phone 2: Redmi 8 [Wi-Fi]")
        ]
        for url, label in wifi_phones:
            sources.append({
                "id": url,
                "name": label,
                "type": "wifi_phone",
                "active": True
            })

        # 3. Virtual Simulation
        sources.append({
            "id": "mock",
            "name": "Virtual Simulation (Offline Test)",
            "type": "mock",
            "active": True
        })

        return sources

    def generate_mjpeg_stream(self, cam_id: int) -> Generator[bytes, None, None]:
        """Generator yielding MJPEG multipart frames for ultra-low latency HTTP streaming."""
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, 55]
        while True:
            frame = self.get_annotated_frame(cam_id)
            if frame is None:
                frame = np.zeros((360, 640, 3), dtype=np.uint8)
                cv2.putText(frame, f"Connecting to Cam {cam_id}...", (160, 180),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)

            ret, jpeg = cv2.imencode('.jpg', frame, encode_params)
            if ret:
                chunk = (
                    b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n'
                )
                yield chunk

            time.sleep(0.025)  # ~40 FPS stream
