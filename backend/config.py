from __future__ import annotations
import os
import yaml
from pathlib import Path
from typing import Dict, Any, List

CONFIG_FILE_PATH = Path(__file__).resolve().parent.parent / "config.yaml"

DEFAULT_CONFIG: Dict[str, Any] = {
    "cameras": {
        "cam1": {
            "id": 1,
            "name": "Logitech 720p (Webcam)",
            "source": 2,
            "players": [1, 2],
            "flip_horizontal": True
        },
        "cam2": {
            "id": 2,
            "name": "Phone 1 (USB ADB / IP Webcam)",
            "source": "http://127.0.0.1:8081/video",
            "players": [3, 4],
            "flip_horizontal": False
        },
        "cam3": {
            "id": 3,
            "name": "Phone 2 (USB ADB / IP Webcam)",
            "source": "http://127.0.0.1:8082/video",
            "players": [5, 6],
            "flip_horizontal": False
        }
    },
    "mock_fallback_enabled": True,
    "players": {
        1: "Player 1",
        2: "Player 2",
        3: "Player 3",
        4: "Player 4",
        5: "Player 5",
        6: "Player 6"
    },
    "gaze_thresholds": {
        "yaw_sideways_deg": 24.0,
        "pitch_down_deg": -18.0,
        "pitch_up_deg": 22.0,
        "facing_away_yaw_deg": 42.0,
        "facing_away_pitch_deg": 32.0
    },
    "reaction_thresholds": {
        "facial_velocity": 0.25,
        "head_angular_velocity": 35.0,
        "posture_delta": 0.28,
        "disappear_duration_sec": 0.35
    },
    "penalty_weights": {
        "away_time": 2.0,
        "look_away_count": 5.0,
        "reaction_intensity": 10.0,
        "disappear_time": 15.0,
        "lost_attention_pct": 1.5
    },
    "match": {
        "total_videos": 40,
        "countdown_seconds": 3,
        "suspense_min_sec": 1.5,
        "suspense_max_sec": 3.5,
        "video_dir": "data/videos"
    },
    "adb": {
        "tools_dir": "tools/platform-tools",
        "phone1_port": 8081,
        "phone2_port": 8082,
        "target_port": 8080
    }
}


def load_config() -> Dict[str, Any]:
    if CONFIG_FILE_PATH.exists():
        try:
            with open(CONFIG_FILE_PATH, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
                # Merge with default config
                merged = DEFAULT_CONFIG.copy()
                for k, v in data.items():
                    if isinstance(v, dict) and k in merged:
                        merged[k] = {**merged[k], **v}
                    else:
                        merged[k] = v
                return merged
        except Exception as e:
            print(f"Error loading config.yaml: {e}. Using defaults.")
    return DEFAULT_CONFIG.copy()


def save_config(new_config: Dict[str, Any]) -> bool:
    try:
        with open(CONFIG_FILE_PATH, "w", encoding="utf-8") as f:
            yaml.safe_dump(new_config, f, sort_keys=False)
        return True
    except Exception as e:
        print(f"Failed to save config: {e}")
        return False
