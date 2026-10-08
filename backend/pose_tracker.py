from __future__ import annotations
import math
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "face_landmarker.task"


class GazeState(str, Enum):
    FACING_TV = "FACING_TV"
    FACING_AWAY = "FACING_AWAY"
    LOOKING_DOWN = "LOOKING_DOWN"
    LOOKING_SIDEWAYS = "LOOKING_SIDEWAYS"
    FACE_NOT_DETECTED = "FACE_NOT_DETECTED"


@dataclass
class PlayerPoseData:
    player_id: int
    slot: str  # "left" or "right"
    face_detected: bool = False
    yaw: float = 0.0          # Smoothed
    pitch: float = 0.0        # Smoothed
    roll: float = 0.0         # Smoothed
    raw_yaw: float = 0.0
    raw_pitch: float = 0.0
    raw_roll: float = 0.0
    calibrated_yaw: float = 0.0
    calibrated_pitch: float = 0.0
    calibrated_roll: float = 0.0
    gaze_state: GazeState = GazeState.FACE_NOT_DETECTED
    mouth_aspect_ratio: float = 0.0
    eye_aspect_ratio: float = 0.0
    iris_x_ratio: float = 0.5   # 0.0 = looking left, 1.0 = looking right, 0.5 = center
    iris_y_ratio: float = 0.5   # 0.0 = looking up, 1.0 = looking down, 0.5 = center
    blendshapes: Dict[str, float] = field(default_factory=dict)
    bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)
    nose_2d: Tuple[int, int] = (0, 0)
    left_iris_2d: Tuple[int, int] = (0, 0)
    right_iris_2d: Tuple[int, int] = (0, 0)
    last_seen: float = 0.0
    time_in_current_state: float = 0.0


# RIGID 3D anthropometric cranial/nasal model (in mm)
# Invariant to jaw opening, laughing, smiling, talking (no mouth or chin points!)
RIGID_3D_MODEL = np.array([
    (0.0, 0.0, 0.0),          # Nose tip (landmark 1)
    (0.0, -18.0, -8.0),       # Nose bottom / subnasale (landmark 4)
    (0.0, 24.0, -16.0),       # Nose bridge / nasion (landmark 6)
    (0.0, 52.0, -26.0),       # Mid-eyebrow / glabella (landmark 168)
    (-52.0, 26.0, -42.0),     # Left eye outer corner (landmark 33)
    (52.0, 26.0, -42.0),      # Right eye outer corner (landmark 263)
    (-76.0, 0.0, -85.0),      # Left tragus / ear anchor (landmark 234)
    (76.0, 0.0, -85.0)        # Right tragus / ear anchor (landmark 454)
], dtype=np.float64)

RIGID_LANDMARK_INDICES = [1, 4, 6, 168, 33, 263, 234, 454]


class AdaptiveAngleFilter:
    """
    Adaptive dual-rate EMA smoothing filter.
    Kills camera sensor jitter and landmark noise at rest (slow_alpha = 0.20),
    while tracking intentional head turns instantly with zero lag (fast_alpha = 0.85).
    """
    def __init__(self, slow_alpha: float = 0.20, fast_alpha: float = 0.85, delta_thresh: float = 10.0):
        self.slow_alpha = slow_alpha
        self.fast_alpha = fast_alpha
        self.delta_thresh = delta_thresh
        self.val: Optional[float] = None

    def update(self, new_val: float) -> float:
        if self.val is None:
            self.val = new_val
            return new_val
        delta = abs(new_val - self.val)
        ratio = min(1.0, max(0.0, delta / self.delta_thresh))
        alpha = (1.0 - ratio) * self.slow_alpha + ratio * self.fast_alpha
        self.val = alpha * new_val + (1.0 - alpha) * self.val
        return self.val

    def reset(self):
        self.val = None


class PoseTracker:
    def __init__(self, model_path: Optional[Path] = None, gaze_thresholds: Optional[Dict[str, float]] = None):
        self.model_path = model_path or MODEL_PATH
        self.gaze_thresholds = gaze_thresholds or {
            "yaw_sideways_deg": 24.0,
            "pitch_down_deg": -18.0,
            "pitch_up_deg": 22.0,
            "facing_away_yaw_deg": 40.0,
            "facing_away_pitch_deg": 30.0,
            "iris_sideways_thresh": 0.20,
            "iris_down_thresh": 0.22
        }

        # Calibration baselines per player_id: {player_id: (yaw0, pitch0, roll0)}
        self.calibration_baselines: Dict[int, Tuple[float, float, float]] = {
            i: (0.0, 0.0, 0.0) for i in range(1, 7)
        }

        # Smoothing filters per player_id
        self.yaw_filters: Dict[int, AdaptiveAngleFilter] = {i: AdaptiveAngleFilter() for i in range(1, 7)}
        self.pitch_filters: Dict[int, AdaptiveAngleFilter] = {i: AdaptiveAngleFilter() for i in range(1, 7)}
        self.roll_filters: Dict[int, AdaptiveAngleFilter] = {i: AdaptiveAngleFilter() for i in range(1, 7)}

        # State tracking for duration and debouncing
        self.previous_states: Dict[int, GazeState] = {
            i: GazeState.FACE_NOT_DETECTED for i in range(1, 7)
        }
        self.state_start_times: Dict[int, float] = {
            i: time.time() for i in range(1, 7)
        }

        # Initialize MediaPipe FaceLandmarker
        if not self.model_path.exists():
            raise FileNotFoundError(f"MediaPipe face landmarker model missing at {self.model_path}")

        base_options = mp_python.BaseOptions(model_asset_path=str(self.model_path))
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
            num_faces=2,
            min_face_detection_confidence=0.45,
            min_face_presence_confidence=0.45,
            min_tracking_confidence=0.45
        )
        self.detector = vision.FaceLandmarker.create_from_options(options)

    def set_calibration(self, player_id: int, yaw: float, pitch: float, roll: float):
        self.calibration_baselines[player_id] = (yaw, pitch, roll)

    def _extract_euler_from_matrix(self, trans_mat: np.ndarray) -> Tuple[float, float, float]:
        """Extracts pitch, yaw, roll (degrees) from 4x4 transformation matrix."""
        R = trans_mat[:3, :3]
        # In MediaPipe metric coordinate system:
        # Pitch: rotation around X axis (nodding up/down)
        # Yaw: rotation around Y axis (turning left/right)
        # Roll: rotation around Z axis (tilting side-to-side)
        pitch = math.degrees(math.atan2(-R[1, 2], R[2, 2]))
        yaw = math.degrees(math.atan2(R[0, 2], math.sqrt(R[1, 2] ** 2 + R[2, 2] ** 2)))
        roll = math.degrees(math.atan2(-R[0, 1], R[0, 0]))
        return yaw, pitch, roll

    def _solve_rigid_pnp(self, landmarks, w: int, h: int, camera_matrix: np.ndarray, dist_coeffs: np.ndarray) -> Tuple[float, float, float]:
        """Rigid cranberry-cranial solvePnP fallback (no chin or mouth points)."""
        image_points = np.array([
            (landmarks[idx].x * w, landmarks[idx].y * h)
            for idx in RIGID_LANDMARK_INDICES
        ], dtype=np.float64)

        success, rot_vec, _ = cv2.solvePnP(
            RIGID_3D_MODEL,
            image_points,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_EPNP
        )
        if success:
            rot_mat, _ = cv2.Rodrigues(rot_vec)
            pitch = math.degrees(math.atan2(rot_mat[2, 1], rot_mat[2, 2]))
            yaw = math.degrees(math.atan2(-rot_mat[2, 0], math.sqrt(rot_mat[2, 1] ** 2 + rot_mat[2, 2] ** 2)))
            roll = math.degrees(math.atan2(rot_mat[1, 0], rot_mat[0, 0]))
            return yaw, pitch, roll
        return 0.0, 0.0, 0.0

    def _compute_iris_gaze(self, landmarks, w: int, h: int) -> Tuple[float, float, float, Tuple[int, int], Tuple[int, int]]:
        """
        Computes normalized iris gaze ratios (horizontal & vertical) and eye aspect ratio.
        Landmarks:
          Left eye:  33 outer, 133 inner, 159 top, 145 bot, 468 iris
          Right eye: 263 outer, 362 inner, 386 top, 374 bot, 473 iris
        """
        has_iris = len(landmarks) >= 478

        # Left eye coordinates
        lx_out, ly_out = landmarks[33].x * w, landmarks[33].y * h
        lx_in, ly_in = landmarks[133].x * w, landmarks[133].y * h
        ly_top = landmarks[159].y * h
        ly_bot = landmarks[145].y * h
        l_iris = (int(landmarks[468].x * w), int(landmarks[468].y * h)) if has_iris else (int((lx_out + lx_in) / 2), int((ly_top + ly_bot) / 2))

        # Right eye coordinates
        rx_out, ry_out = landmarks[263].x * w, landmarks[263].y * h
        rx_in, ry_in = landmarks[362].x * w, landmarks[362].y * h
        ry_top = landmarks[386].y * h
        ry_bot = landmarks[374].y * h
        r_iris = (int(landmarks[473].x * w), int(landmarks[473].y * h)) if has_iris else (int((rx_out + rx_in) / 2), int((ry_top + ry_bot) / 2))

        # Horizontal ratio (0.0 = looking camera-left, 1.0 = looking camera-right)
        l_span = max(1.0, abs(lx_out - lx_in))
        r_span = max(1.0, abs(rx_out - rx_in))
        lh_ratio = (l_iris[0] - min(lx_out, lx_in)) / l_span
        rh_ratio = (r_iris[0] - min(rx_out, rx_in)) / r_span
        h_ratio = float(np.clip((lh_ratio + rh_ratio) / 2.0, 0.0, 1.0))

        # Vertical ratio (0.0 = looking up, 1.0 = looking down)
        lv_span = max(1.0, abs(ly_bot - ly_top))
        rv_span = max(1.0, abs(ry_bot - ry_top))
        lv_ratio = (l_iris[1] - min(ly_top, ly_bot)) / lv_span
        rv_ratio = (r_iris[1] - min(ry_top, ry_bot)) / rv_span
        v_ratio = float(np.clip((lv_ratio + rv_ratio) / 2.0, 0.0, 1.0))

        # Eye Aspect Ratio (EAR) for squint / eye blink
        ear = float(((lv_span / l_span) + (rv_span / r_span)) / 2.0)

        return h_ratio, v_ratio, ear, l_iris, r_iris

    def process_camera_frame(
        self,
        frame: np.ndarray,
        player_left_id: int,
        player_right_id: int
    ) -> Tuple[PlayerPoseData, PlayerPoseData]:
        h, w = frame.shape[:2]
        now = time.time()

        # Create MediaPipe image
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # Detect faces
        detection_result = self.detector.detect(mp_image)

        left_player = PlayerPoseData(player_id=player_left_id, slot="left", last_seen=0.0)
        right_player = PlayerPoseData(player_id=player_right_id, slot="right", last_seen=0.0)

        # Intrinsic Camera matrix approximation
        focal_length = w
        center = (w / 2.0, h / 2.0)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype=np.float64)
        dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        if detection_result.face_landmarks:
            num_detected = len(detection_result.face_landmarks)
            for face_idx in range(num_detected):
                landmarks = detection_result.face_landmarks[face_idx]

                # Compute centroid and bbox
                xs = [lm.x * w for lm in landmarks]
                ys = [lm.y * h for lm in landmarks]
                min_x, max_x = max(0, int(min(xs))), min(w, int(max(xs)))
                min_y, max_y = max(0, int(min(ys))), min(h, int(max(ys)))
                cx = (min_x + max_x) / 2.0

                # Determine slot assignment
                target_player = left_player if cx < (w / 2.0) else right_player

                # 1. Pose estimation: Try facial transformation matrix first, fallback to rigid solvePnP
                raw_yaw, raw_pitch, raw_roll = 0.0, 0.0, 0.0
                if detection_result.facial_transformation_matrixes and face_idx < len(detection_result.facial_transformation_matrixes):
                    trans_mat = np.array(detection_result.facial_transformation_matrixes[face_idx])
                    raw_yaw, raw_pitch, raw_roll = self._extract_euler_from_matrix(trans_mat)
                else:
                    raw_yaw, raw_pitch, raw_roll = self._solve_rigid_pnp(landmarks, w, h, camera_matrix, dist_coeffs)

                # 2. Adaptive EMA smoothing (eliminates jitter completely!)
                pid = target_player.player_id
                smooth_yaw = self.yaw_filters[pid].update(raw_yaw)
                smooth_pitch = self.pitch_filters[pid].update(raw_pitch)
                smooth_roll = self.roll_filters[pid].update(raw_roll)

                # 3. Iris & Eye Gaze analysis
                iris_h, iris_v, ear, l_iris, r_iris = self._compute_iris_gaze(landmarks, w, h)

                # 4. Blendshapes extraction
                blendshape_dict = {}
                if detection_result.face_blendshapes and face_idx < len(detection_result.face_blendshapes):
                    for cat in detection_result.face_blendshapes[face_idx]:
                        blendshape_dict[cat.category_name] = cat.score

                # 5. Mouth Aspect Ratio (MAR)
                lip_top = np.array([landmarks[13].x * w, landmarks[13].y * h])
                lip_bot = np.array([landmarks[14].x * w, landmarks[14].y * h])
                lip_left = np.array([landmarks[61].x * w, landmarks[61].y * h])
                lip_right = np.array([landmarks[291].x * w, landmarks[291].y * h])
                vertical_dist = np.linalg.norm(lip_top - lip_bot)
                horizontal_dist = max(1.0, np.linalg.norm(lip_left - lip_right))
                mar = float(vertical_dist / horizontal_dist)

                # Populate target player
                target_player.face_detected = True
                target_player.yaw = round(smooth_yaw, 2)
                target_player.pitch = round(smooth_pitch, 2)
                target_player.roll = round(smooth_roll, 2)
                target_player.raw_yaw = round(raw_yaw, 2)
                target_player.raw_pitch = round(raw_pitch, 2)
                target_player.raw_roll = round(raw_roll, 2)
                target_player.bbox = (min_x, min_y, max_x, max_y)
                target_player.nose_2d = (int(landmarks[1].x * w), int(landmarks[1].y * h))
                target_player.left_iris_2d = l_iris
                target_player.right_iris_2d = r_iris
                target_player.iris_x_ratio = round(iris_h, 3)
                target_player.iris_y_ratio = round(iris_v, 3)
                target_player.eye_aspect_ratio = round(ear, 3)
                target_player.mouth_aspect_ratio = round(mar, 3)
                target_player.blendshapes = blendshape_dict
                target_player.last_seen = now

        # Classify gaze states and compute calibrated angles for both players
        for player in [left_player, right_player]:
            pid = player.player_id
            b_yaw, b_pitch, b_roll = self.calibration_baselines.get(pid, (0.0, 0.0, 0.0))

            if player.face_detected:
                c_yaw = player.yaw - b_yaw
                c_pitch = player.pitch - b_pitch
                c_roll = player.roll - b_roll
                player.calibrated_yaw = round(c_yaw, 2)
                player.calibrated_pitch = round(c_pitch, 2)
                player.calibrated_roll = round(c_roll, 2)

                # Classify Gaze State using both Head Pose + Eye Iris
                away_yaw_thresh = self.gaze_thresholds.get("facing_away_yaw_deg", 40.0)
                away_pitch_thresh = self.gaze_thresholds.get("facing_away_pitch_deg", 30.0)
                side_yaw_thresh = self.gaze_thresholds.get("yaw_sideways_deg", 24.0)
                down_pitch_thresh = self.gaze_thresholds.get("pitch_down_deg", -18.0)
                iris_side_thresh = self.gaze_thresholds.get("iris_sideways_thresh", 0.20)
                iris_down_thresh = self.gaze_thresholds.get("iris_down_thresh", 0.22)

                # Calculate iris deviation from neutral center (0.5)
                iris_side_offset = abs(player.iris_x_ratio - 0.50)
                iris_down_offset = max(0.0, player.iris_y_ratio - 0.50)

                # 1. FACING_AWAY (extreme turn)
                if abs(c_yaw) > away_yaw_thresh or abs(c_pitch) > away_pitch_thresh:
                    player.gaze_state = GazeState.FACING_AWAY
                # 2. LOOKING_DOWN (head pitched down OR eyes glanced down)
                elif c_pitch < down_pitch_thresh or (iris_down_offset > iris_down_thresh and c_pitch < -8.0):
                    player.gaze_state = GazeState.LOOKING_DOWN
                # 3. LOOKING_SIDEWAYS (head turned sideways OR eyes turned sideways)
                elif abs(c_yaw) > side_yaw_thresh or (iris_side_offset > iris_side_thresh and abs(c_yaw) > 12.0):
                    player.gaze_state = GazeState.LOOKING_SIDEWAYS
                # 4. FACING_TV
                else:
                    player.gaze_state = GazeState.FACING_TV
            else:
                player.gaze_state = GazeState.FACE_NOT_DETECTED
                # Reset filters when face disappears
                self.yaw_filters[pid].reset()
                self.pitch_filters[pid].reset()
                self.roll_filters[pid].reset()

            # Track time in current state
            prev_st = self.previous_states[pid]
            if player.gaze_state != prev_st:
                self.previous_states[pid] = player.gaze_state
                self.state_start_times[pid] = now

            player.time_in_current_state = round(now - self.state_start_times[pid], 2)

        return left_player, right_player

    def draw_hud(
        self,
        frame: np.ndarray,
        left_player: PlayerPoseData,
        right_player: PlayerPoseData,
        player_names: Optional[Dict[int, str]] = None
    ) -> np.ndarray:
        """Renders live broadcast HUD overlays onto the camera frame."""
        h, w = frame.shape[:2]
        vis = frame.copy()

        # Slot Divider Line
        cv2.line(vis, (w // 2, 0), (w // 2, h), (70, 70, 70), 1, cv2.LINE_AA)

        # State Colors (BGR)
        state_colors = {
            GazeState.FACING_TV: (0, 220, 80),          # Vibrant Green
            GazeState.LOOKING_DOWN: (0, 165, 255),       # Amber / Orange
            GazeState.LOOKING_SIDEWAYS: (0, 200, 255),   # Yellow
            GazeState.FACING_AWAY: (40, 40, 255),        # Red
            GazeState.FACE_NOT_DETECTED: (130, 130, 130) # Gray
        }

        for player in [left_player, right_player]:
            color = state_colors.get(player.gaze_state, (200, 200, 200))
            name = (player_names or {}).get(player.player_id, f"Player {player.player_id}")

            # Base X position for player slot HUD text
            slot_x = 15 if player.slot == "left" else (w // 2 + 15)

            # Player Banner Card
            cv2.rectangle(vis, (slot_x - 5, 10), (slot_x + 240, 84), (20, 20, 25), -1)
            cv2.rectangle(vis, (slot_x - 5, 10), (slot_x + 240, 84), color, 2)

            # Player Name
            cv2.putText(vis, name, (slot_x, 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

            # State Pill
            cv2.putText(vis, player.gaze_state.value, (slot_x, 54),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, color, 2, cv2.LINE_AA)

            # Yaw / Pitch / Roll Telemetry
            if player.face_detected:
                angles_text = f"Y:{player.calibrated_yaw:+.1f} P:{player.calibrated_pitch:+.1f} R:{player.calibrated_roll:+.1f}"
                cv2.putText(vis, angles_text, (slot_x, 74),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.40, (180, 180, 180), 1, cv2.LINE_AA)

                # Draw Face Bounding Box
                x1, y1, x2, y2 = player.bbox
                cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

                # Draw Nose Center Point
                nx, ny = player.nose_2d
                cv2.circle(vis, (nx, ny), 3, (0, 255, 255), -1)

                # Draw Iris tracking dots if available
                if player.left_iris_2d != (0, 0):
                    cv2.circle(vis, player.left_iris_2d, 3, (255, 255, 0), -1)
                if player.right_iris_2d != (0, 0):
                    cv2.circle(vis, player.right_iris_2d, 3, (255, 255, 0), -1)

                # Draw 3D Orientation Vector (combined head + iris gaze line)
                gaze_len = 50.0
                rad_yaw = math.radians(player.calibrated_yaw + (player.iris_x_ratio - 0.5) * 40.0)
                rad_pitch = math.radians(player.calibrated_pitch - (player.iris_y_ratio - 0.5) * 30.0)
                end_x = int(nx + gaze_len * math.sin(rad_yaw))
                end_y = int(ny - gaze_len * math.sin(rad_pitch))
                cv2.arrowedLine(vis, (nx, ny), (end_x, end_y), color, 2, tipLength=0.25)

            else:
                cv2.putText(vis, "FACE MISSING", (slot_x, 74),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (120, 120, 120), 1, cv2.LINE_AA)

        return vis
