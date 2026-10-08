from __future__ import annotations
import json
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Any, Optional

from backend.pose_tracker import PlayerPoseData, GazeState
from backend.reaction_detector import ReactionEvent, EventType
from backend.video_manager import VideoManager

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MATCHES_DIR = PROJECT_ROOT / "data" / "matches"


class MatchState(str, Enum):
    LOBBY = "LOBBY"
    CALIBRATING = "CALIBRATING"
    COUNTDOWN = "COUNTDOWN"
    PLAYING_VIDEO = "PLAYING_VIDEO"
    SUSPENSE_INTERVAL = "SUSPENSE_INTERVAL"
    MATCH_OVER = "MATCH_OVER"


@dataclass
class PlayerSessionStats:
    player_id: int
    name: str
    total_facing_tv_sec: float = 0.0
    total_away_sec: float = 0.0
    total_disappeared_sec: float = 0.0
    total_face_detected_sec: float = 0.0
    look_away_count: int = 0
    current_look_streak_sec: float = 0.0
    longest_look_streak_sec: float = 0.0
    reaction_events: List[Dict[str, Any]] = field(default_factory=list)
    penalty_score: float = 0.0
    tv_attention_pct: float = 100.0
    rank: int = 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "player_id": self.player_id,
            "name": self.name,
            "total_facing_tv_sec": round(self.total_facing_tv_sec, 2),
            "total_away_sec": round(self.total_away_sec, 2),
            "total_disappeared_sec": round(self.total_disappeared_sec, 2),
            "total_face_detected_sec": round(self.total_face_detected_sec, 2),
            "look_away_count": self.look_away_count,
            "current_look_streak_sec": round(self.current_look_streak_sec, 2),
            "longest_look_streak_sec": round(self.longest_look_streak_sec, 2),
            "reaction_count": len(self.reaction_events),
            "penalty_score": round(self.penalty_score, 1),
            "tv_attention_pct": round(self.tv_attention_pct, 1),
            "rank": self.rank,
            "recent_events": self.reaction_events[-5:]
        }


class GameEngine:
    def __init__(self, config: Dict[str, Any], video_manager: VideoManager):
        self.config = config
        self.video_manager = video_manager
        self.match_state = MatchState.LOBBY
        self.match_start_time: float = 0.0
        self.state_start_time: float = time.time()
        self.active_play_time: float = 0.0
        self.last_tick_time: float = time.time()

        # Calibration state
        self.calibration_duration: float = 3.0
        self.calibration_samples: Dict[int, List[tuple]] = {i: [] for i in range(1, 7)}

        # Video progress
        self.current_round: int = 1
        self.total_rounds: int = config.get("match", {}).get("total_videos", 40)
        self.countdown_seconds: int = config.get("match", {}).get("countdown_seconds", 3)
        self.current_suspense_duration: float = 2.0

        # Player stats
        self.player_stats: Dict[int, PlayerSessionStats] = {}
        self.reset_player_stats()

        MATCHES_DIR.mkdir(parents=True, exist_ok=True)

    def reset_player_stats(self):
        player_names = self.config.get("players", {})
        self.player_stats = {
            i: PlayerSessionStats(
                player_id=i,
                name=player_names.get(i, f"Player {i}")
            ) for i in range(1, 7)
        }

    def start_match(self):
        self.video_manager.start_new_match_playlist()
        self.reset_player_stats()
        self.current_round = 1
        self.active_play_time = 0.0
        self.match_start_time = time.time()
        self.transition_to(MatchState.COUNTDOWN)

    def stop_match(self):
        self.transition_to(MatchState.LOBBY)

    def start_calibration(self):
        self.calibration_samples = {i: [] for i in range(1, 7)}
        self.transition_to(MatchState.CALIBRATING)

    def transition_to(self, new_state: MatchState):
        self.match_state = new_state
        self.state_start_time = time.time()
        if new_state == MatchState.SUSPENSE_INTERVAL:
            import random
            s_min = self.config.get("match", {}).get("suspense_min_sec", 1.5)
            s_max = self.config.get("match", {}).get("suspense_max_sec", 3.5)
            self.current_suspense_duration = random.uniform(s_min, s_max)

    def update_calibration_samples(self, player_poses: Dict[int, PlayerPoseData]):
        for pid, pose in player_poses.items():
            if pose.face_detected:
                self.calibration_samples[pid].append((pose.yaw, pose.pitch, pose.roll))

    def finish_calibration(self) -> Dict[int, tuple]:
        baselines = {}
        for pid, samples in self.calibration_samples.items():
            if samples:
                avg_y = sum(s[0] for s in samples) / len(samples)
                avg_p = sum(s[1] for s in samples) / len(samples)
                avg_r = sum(s[2] for s in samples) / len(samples)
                baselines[pid] = (round(avg_y, 2), round(avg_p, 2), round(avg_r, 2))
            else:
                baselines[pid] = (0.0, 0.0, 0.0)
        self.transition_to(MatchState.LOBBY)
        return baselines

    def tick(
        self,
        player_poses: Dict[int, PlayerPoseData],
        new_events: List[ReactionEvent]
    ) -> Dict[str, Any]:
        now = time.time()
        dt = max(0.0001, min(0.2, now - self.last_tick_time))
        self.last_tick_time = now

        # Handle State Timers
        if self.match_state == MatchState.CALIBRATING:
            elapsed = now - self.state_start_time
            self.update_calibration_samples(player_poses)
            if elapsed >= self.calibration_duration:
                # Finished calibration
                baselines = self.finish_calibration()
                return {"calibration_finished": True, "baselines": baselines}

        elif self.match_state == MatchState.COUNTDOWN:
            elapsed = now - self.state_start_time
            if elapsed >= self.countdown_seconds:
                self.transition_to(MatchState.PLAYING_VIDEO)

        elif self.match_state == MatchState.PLAYING_VIDEO:
            self.active_play_time += dt
            self._accumulate_player_telemetry(player_poses, new_events, dt)

        elif self.match_state == MatchState.SUSPENSE_INTERVAL:
            elapsed = now - self.state_start_time
            if elapsed >= self.current_suspense_duration:
                # Next video or Match Over
                next_vid = self.video_manager.next_video()
                if next_vid:
                    self.current_round += 1
                    self.transition_to(MatchState.PLAYING_VIDEO)
                else:
                    self.transition_to(MatchState.MATCH_OVER)
                    self.save_match_summary()

        self._recalculate_penalties_and_ranks()
        return self.get_game_state_payload()

    def on_video_finished(self):
        """Called by the TV video player when a clip ends."""
        if self.match_state == MatchState.PLAYING_VIDEO:
            if self.current_round >= self.total_rounds:
                self.transition_to(MatchState.MATCH_OVER)
                self.save_match_summary()
            else:
                self.transition_to(MatchState.SUSPENSE_INTERVAL)

    def _accumulate_player_telemetry(
        self,
        player_poses: Dict[int, PlayerPoseData],
        new_events: List[ReactionEvent],
        dt: float
    ):
        for pid, pose in player_poses.items():
            stats = self.player_stats[pid]

            # Face detection duration
            if pose.face_detected:
                stats.total_face_detected_sec += dt

            # Gaze state duration
            if pose.gaze_state == GazeState.FACING_TV:
                stats.total_facing_tv_sec += dt
                stats.current_look_streak_sec += dt
                if stats.current_look_streak_sec > stats.longest_look_streak_sec:
                    stats.longest_look_streak_sec = stats.current_look_streak_sec
            else:
                stats.current_look_streak_sec = 0.0
                stats.total_away_sec += dt
                if pose.gaze_state == GazeState.FACE_NOT_DETECTED:
                    stats.total_disappeared_sec += dt

        # Append new events
        for ev in new_events:
            stats = self.player_stats.get(ev.player_id)
            if stats:
                stats.reaction_events.append(ev.to_dict())
                if ev.event_type == EventType.LOOK_AWAY:
                    stats.look_away_count += 1

    def _recalculate_penalties_and_ranks(self):
        w = self.config.get("penalty_weights", {
            "away_time": 2.0,
            "look_away_count": 5.0,
            "reaction_intensity": 10.0,
            "disappear_time": 15.0,
            "lost_attention_pct": 1.5
        })

        for pid, stats in self.player_stats.items():
            # TV Attention %
            if self.active_play_time > 0:
                stats.tv_attention_pct = min(100.0, max(0.0, (stats.total_facing_tv_sec / self.active_play_time) * 100.0))
            else:
                stats.tv_attention_pct = 100.0

            # Sum of reaction intensities
            intensity_sum = sum(e.get("intensity", 0.0) for e in stats.reaction_events)

            # Penalty score formula
            lost_attention = 100.0 - stats.tv_attention_pct
            stats.penalty_score = (
                (w.get("away_time", 2.0) * stats.total_away_sec) +
                (w.get("look_away_count", 5.0) * stats.look_away_count) +
                (w.get("reaction_intensity", 10.0) * intensity_sum) +
                (w.get("disappear_time", 15.0) * stats.total_disappeared_sec) +
                (w.get("lost_attention_pct", 1.5) * lost_attention)
            )

        # Rank players: Lowest penalty score = 1st place (best)
        sorted_players = sorted(self.player_stats.values(), key=lambda p: p.penalty_score)
        for rank_idx, player in enumerate(sorted_players):
            player.rank = rank_idx + 1

    def get_game_state_payload(self) -> Dict[str, Any]:
        curr_vid = self.video_manager.get_current_video()
        return {
            "match_state": self.match_state.value,
            "current_round": self.current_round,
            "total_rounds": self.total_rounds,
            "active_play_time": round(self.active_play_time, 2),
            "state_elapsed": round(time.time() - self.state_start_time, 2),
            "current_video": curr_vid,
            "leaderboard": [p.to_dict() for p in sorted(self.player_stats.values(), key=lambda x: x.penalty_score)]
        }

    def save_match_summary(self) -> str:
        timestamp_str = time.strftime("%Y%m%d_%H%M%S")
        filename = f"match_{timestamp_str}.json"
        filepath = MATCHES_DIR / filename
        summary = {
            "match_id": timestamp_str,
            "timestamp": time.time(),
            "total_active_time_sec": round(self.active_play_time, 2),
            "total_rounds": self.total_rounds,
            "final_leaderboard": [p.to_dict() for p in sorted(self.player_stats.values(), key=lambda x: x.penalty_score)],
            "full_event_history": [
                event
                for p in self.player_stats.values()
                for event in p.reaction_events
            ]
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"Match report saved to {filepath}")
        return str(filepath)
