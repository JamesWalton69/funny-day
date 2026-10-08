from __future__ import annotations
import math
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Deque, Tuple
from collections import deque

from backend.pose_tracker import PlayerPoseData, GazeState


class EventType(str, Enum):
    FACIAL_MOVEMENT = "SUDDEN_FACIAL_MOVEMENT"
    HEAD_MOVEMENT = "SUDDEN_HEAD_MOVEMENT"
    POSTURE_MOVEMENT = "SUDDEN_POSTURE_MOVEMENT"
    LOOK_AWAY = "LOOKED_AWAY"
    DISAPPEARED = "DISAPPEARED_FROM_CAMERA"


@dataclass
class ReactionEvent:
    event_id: str
    player_id: int
    timestamp: float
    game_time_sec: float
    video_round: int
    event_type: EventType
    intensity: float  # Normalized 0.0 to 1.0
    description: str

    def to_dict(self) -> Dict:
        return {
            "event_id": self.event_id,
            "player_id": self.player_id,
            "timestamp": round(self.timestamp, 3),
            "game_time_sec": round(self.game_time_sec, 2),
            "video_round": self.video_round,
            "event_type": self.event_type.value,
            "intensity": round(self.intensity, 3),
            "description": self.description
        }


class ReactionDetector:
    """
    Temporal multi-scale reaction detector.
    Analyzes true human neurological reaction onsets (120ms - 300ms window),
    eliminating 1-frame division-by-dt sensor noise spikes while maintaining
    high sensitivity to smiles, laughs, gasps, flinches, and look-aways.
    """
    def __init__(self, thresholds: Optional[Dict[str, float]] = None):
        self.thresholds = thresholds or {
            "facial_velocity": 0.25,
            "smile_onset_thresh": 0.22,
            "smile_absolute_thresh": 0.40,
            "gasp_onset_thresh": 0.25,
            "head_flinch_angle_deg": 18.0,
            "posture_shift_thresh": 0.22,
            "look_away_debounce_sec": 0.18,
            "disappear_duration_sec": 0.35
        }

        # Sliding temporal buffer per player: deque of (timestamp, PlayerPoseData)
        # Keeps last 1.0 second of data (~30-60 frames)
        self.history: Dict[int, Deque[Tuple[float, PlayerPoseData]]] = {
            i: deque(maxlen=40) for i in range(1, 7)
        }

        # Event cooldown timestamps: { (player_id, EventType): last_trigger_time }
        self.last_triggered: Dict[Tuple[int, EventType], float] = {}

        # Look away timer: { player_id: (first_away_time, gaze_state) }
        self.away_tracker: Dict[int, Optional[Tuple[float, GazeState]]] = {i: None for i in range(1, 7)}

        # Disappearance timer: { player_id: first_lost_time }
        self.disappear_start: Dict[int, Optional[float]] = {i: None for i in range(1, 7)}

    def _get_historical_pose(self, pid: int, target_age_sec: float) -> Optional[PlayerPoseData]:
        """Finds the pose closest to target_age_sec in history (e.g. 150ms ago)."""
        hist = self.history[pid]
        if not hist:
            return None
        now = time.time()
        best_pose = None
        min_diff = float("inf")
        for ts, pose in hist:
            age = now - ts
            diff = abs(age - target_age_sec)
            if diff < min_diff:
                min_diff = diff
                best_pose = pose
        return best_pose

    def check_player_reactions(
        self,
        current_pose: PlayerPoseData,
        game_time_sec: float = 0.0,
        current_video_round: int = 1
    ) -> List[ReactionEvent]:
        pid = current_pose.player_id
        now = time.time()
        events: List[ReactionEvent] = []

        hist = self.history[pid]
        hist.append((now, current_pose))

        # -------------------------------------------------------------
        # 1. DISAPPEARED CHECK (Face left camera frame)
        # -------------------------------------------------------------
        if not current_pose.face_detected:
            if self.disappear_start[pid] is None:
                self.disappear_start[pid] = now
            else:
                missing_dur = now - self.disappear_start[pid]
                cooldown_key = (pid, EventType.DISAPPEARED)
                last_time = self.last_triggered.get(cooldown_key, 0.0)

                if missing_dur >= self.thresholds["disappear_duration_sec"] and (now - last_time > 1.2):
                    intensity = min(1.0, 0.5 + (missing_dur / 3.0))
                    ev = ReactionEvent(
                        event_id=str(uuid.uuid4())[:8],
                        player_id=pid,
                        timestamp=now,
                        game_time_sec=game_time_sec,
                        video_round=current_video_round,
                        event_type=EventType.DISAPPEARED,
                        intensity=intensity,
                        description=f"Face disappeared from frame ({missing_dur:.2f}s)"
                    )
                    events.append(ev)
                    self.last_triggered[cooldown_key] = now
            return events
        else:
            self.disappear_start[pid] = None

        # Need at least 2 frames in history for temporal reactions
        if len(hist) < 2:
            return events

        # Reference pose from ~150ms ago (human reaction window)
        ref_pose = self._get_historical_pose(pid, target_age_sec=0.15) or hist[0][1]

        # -------------------------------------------------------------
        # 2. LOOKING AWAY CHECK (Transition from TV)
        # -------------------------------------------------------------
        is_away = current_pose.gaze_state in {
            GazeState.LOOKING_DOWN,
            GazeState.LOOKING_SIDEWAYS,
            GazeState.FACING_AWAY
        }
        was_facing_tv = ref_pose.gaze_state == GazeState.FACING_TV

        if is_away:
            if was_facing_tv:
                cooldown_key = (pid, EventType.LOOK_AWAY)
                last_time = self.last_triggered.get(cooldown_key, 0.0)

                if now - last_time > 0.6:
                    angle_dev = max(abs(current_pose.calibrated_yaw), abs(current_pose.calibrated_pitch))
                    intensity = min(1.0, max(0.35, angle_dev / 50.0))
                    ev = ReactionEvent(
                        event_id=str(uuid.uuid4())[:8],
                        player_id=pid,
                        timestamp=now,
                        game_time_sec=game_time_sec,
                        video_round=current_video_round,
                        event_type=EventType.LOOK_AWAY,
                        intensity=intensity,
                        description=f"Looked away from TV ({current_pose.gaze_state.value})"
                    )
                    events.append(ev)
                    self.last_triggered[cooldown_key] = now
        else:
            self.away_tracker[pid] = None

        # -------------------------------------------------------------
        # 3. FACIAL REACTION (Smiles, Laughs, Gasps, Jaw Drops)
        # -------------------------------------------------------------
        triggered_facial = False
        facial_intensity = 0.0
        facial_desc = ""

        # A) Blendshape expression analysis
        if current_pose.blendshapes:
            # Smile & Laughter
            cur_smile = max(current_pose.blendshapes.get("mouthSmileLeft", 0.0), current_pose.blendshapes.get("mouthSmileRight", 0.0))
            ref_smile = max(ref_pose.blendshapes.get("mouthSmileLeft", 0.0), ref_pose.blendshapes.get("mouthSmileRight", 0.0))
            cur_jaw = current_pose.blendshapes.get("jawOpen", 0.0)
            ref_jaw = ref_pose.blendshapes.get("jawOpen", 0.0)

            cur_laugh = cur_smile * 0.70 + cur_jaw * 0.30
            ref_laugh = ref_smile * 0.70 + ref_jaw * 0.30
            laugh_onset = cur_laugh - ref_laugh

            # Gasp & Surprise
            cur_brow = current_pose.blendshapes.get("browInnerUp", 0.0)
            ref_brow = ref_pose.blendshapes.get("browInnerUp", 0.0)
            cur_wide = max(current_pose.blendshapes.get("eyeWideLeft", 0.0), current_pose.blendshapes.get("eyeWideRight", 0.0))
            ref_wide = max(ref_pose.blendshapes.get("eyeWideLeft", 0.0), ref_pose.blendshapes.get("eyeWideRight", 0.0))

            cur_gasp = cur_brow * 0.60 + cur_wide * 0.40
            ref_gasp = ref_brow * 0.60 + ref_wide * 0.40
            gasp_onset = cur_gasp - ref_gasp

            if cur_laugh >= self.thresholds["smile_absolute_thresh"] or laugh_onset >= self.thresholds["smile_onset_thresh"]:
                triggered_facial = True
                facial_intensity = min(1.0, max(0.4, cur_laugh))
                facial_desc = f"Laughter / Smile detected (Intensity: {facial_intensity:.2f})"
            elif cur_gasp >= 0.42 or gasp_onset >= self.thresholds["gasp_onset_thresh"]:
                triggered_facial = True
                facial_intensity = min(1.0, max(0.4, cur_gasp))
                facial_desc = f"Shock / Gasp detected (Intensity: {facial_intensity:.2f})"

        # B) Mouth Aspect Ratio (MAR) onset fallback
        mar_diff = current_pose.mouth_aspect_ratio - ref_pose.mouth_aspect_ratio
        if not triggered_facial and mar_diff > 0.22:
            triggered_facial = True
            facial_intensity = min(1.0, max(0.35, mar_diff * 1.8))
            facial_desc = f"Sudden mouth movement / jaw drop (Intensity: {facial_intensity:.2f})"

        if triggered_facial:
            cooldown_key = (pid, EventType.FACIAL_MOVEMENT)
            last_time = self.last_triggered.get(cooldown_key, 0.0)
            if now - last_time > 0.6:
                ev = ReactionEvent(
                    event_id=str(uuid.uuid4())[:8],
                    player_id=pid,
                    timestamp=now,
                    game_time_sec=game_time_sec,
                    video_round=current_video_round,
                    event_type=EventType.FACIAL_MOVEMENT,
                    intensity=facial_intensity,
                    description=facial_desc
                )
                events.append(ev)
                self.last_triggered[cooldown_key] = now

        # -------------------------------------------------------------
        # 4. SUDDEN HEAD FLINCH / RECOIL (Compared across 150ms window)
        # -------------------------------------------------------------
        dyaw = current_pose.calibrated_yaw - ref_pose.calibrated_yaw
        dpitch = current_pose.calibrated_pitch - ref_pose.calibrated_pitch
        droll = current_pose.calibrated_roll - ref_pose.calibrated_roll
        angular_delta = math.sqrt(dyaw ** 2 + dpitch ** 2 + droll ** 2)

        if angular_delta >= self.thresholds["head_flinch_angle_deg"]:
            cooldown_key = (pid, EventType.HEAD_MOVEMENT)
            last_time = self.last_triggered.get(cooldown_key, 0.0)
            if now - last_time > 0.6:
                intensity = min(1.0, max(0.4, angular_delta / 45.0))
                ev = ReactionEvent(
                    event_id=str(uuid.uuid4())[:8],
                    player_id=pid,
                    timestamp=now,
                    game_time_sec=game_time_sec,
                    video_round=current_video_round,
                    event_type=EventType.HEAD_MOVEMENT,
                    intensity=intensity,
                    description=f"Sudden head flinch / jerk ({angular_delta:.1f}° shift)"
                )
                events.append(ev)
                self.last_triggered[cooldown_key] = now

        # -------------------------------------------------------------
        # 5. SUDDEN POSTURE MOVEMENT (Ducking, Jumping, Torso Shift)
        # -------------------------------------------------------------
        rx1, ry1, rx2, ry2 = ref_pose.bbox
        cx1, cy1, cx2, cy2 = current_pose.bbox
        if rx2 > rx1 and cx2 > cx1:
            r_cx, r_cy = (rx1 + rx2) / 2.0, (ry1 + ry2) / 2.0
            c_cx, c_cy = (cx1 + cx2) / 2.0, (cy1 + cy2) / 2.0
            shift = math.hypot(c_cx - r_cx, c_cy - r_cy)
            norm_shift = shift / max(1.0, float(cx2 - cx1))

            if norm_shift >= self.thresholds["posture_shift_thresh"]:
                cooldown_key = (pid, EventType.POSTURE_MOVEMENT)
                last_time = self.last_triggered.get(cooldown_key, 0.0)
                if now - last_time > 0.8:
                    intensity = min(1.0, max(0.35, norm_shift * 2.0))
                    ev = ReactionEvent(
                        event_id=str(uuid.uuid4())[:8],
                        player_id=pid,
                        timestamp=now,
                        game_time_sec=game_time_sec,
                        video_round=current_video_round,
                        event_type=EventType.POSTURE_MOVEMENT,
                        intensity=intensity,
                        description=f"Sudden body / posture shift (Intensity: {intensity:.2f})"
                    )
                    events.append(ev)
                    self.last_triggered[cooldown_key] = now

        return events
