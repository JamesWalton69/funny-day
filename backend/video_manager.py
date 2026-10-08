from __future__ import annotations
import os
import random
import time
import math
from pathlib import Path
from typing import List, Dict, Any, Optional
import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_VIDEO_DIR = PROJECT_ROOT / "data" / "videos"
SUPPORTED_EXTENSIONS = {".mp4", ".webm", ".mkv", ".avi", ".mov"}


class VideoManager:
    def __init__(self, video_dir: Optional[Path] = None, target_count: int = 40):
        self.video_dir = video_dir or DEFAULT_VIDEO_DIR
        self.video_dir.mkdir(parents=True, exist_ok=True)
        self.target_count = target_count
        self.playlist: List[Dict[str, Any]] = []
        self.current_index: int = 0
        self.current_video: Optional[Dict[str, Any]] = None

    def refresh_and_ensure_videos(self) -> List[Dict[str, Any]]:
        existing = self.get_available_videos()
        if len(existing) < self.target_count:
            needed = self.target_count - len(existing)
            print(f"Only {len(existing)} challenge videos found. Generating {needed} procedural test challenge clips...")
            self.generate_synthetic_videos(count=needed, start_index=len(existing) + 1)
            existing = self.get_available_videos()
        return existing

    def get_available_videos(self) -> List[Dict[str, Any]]:
        files = []
        for file in sorted(self.video_dir.iterdir()):
            if file.is_file() and file.suffix.lower() in SUPPORTED_EXTENSIONS:
                files.append({
                    "id": file.stem,
                    "filename": file.name,
                    "path": str(file),
                    "url": f"/videos/{file.name}",
                    "size_mb": round(file.stat().st_size / (1024 * 1024), 2)
                })
        return files

    def start_new_match_playlist(self) -> List[Dict[str, Any]]:
        videos = self.refresh_and_ensure_videos()
        if not videos:
            raise RuntimeError("No videos available to start challenge match.")

        # If more than target_count, pick target_count at random
        if len(videos) >= self.target_count:
            selected = random.sample(videos, self.target_count)
        else:
            # Repeat to reach 40 if needed
            selected = (videos * ((self.target_count // len(videos)) + 1))[:self.target_count]
            random.shuffle(selected)

        self.playlist = []
        for idx, item in enumerate(selected):
            self.playlist.append({
                "round_num": idx + 1,
                "total_rounds": self.target_count,
                "video_id": item["id"],
                "filename": item["filename"],
                "url": item["url"],
                "path": item["path"]
            })

        self.current_index = 0
        self.current_video = self.playlist[0] if self.playlist else None
        return self.playlist

    def next_video(self) -> Optional[Dict[str, Any]]:
        if self.current_index + 1 < len(self.playlist):
            self.current_index += 1
            self.current_video = self.playlist[self.current_index]
            return self.current_video
        self.current_video = None
        return None

    def get_current_video(self) -> Optional[Dict[str, Any]]:
        return self.current_video

    def generate_synthetic_videos(self, count: int = 40, start_index: int = 1):
        """Generates engaging procedural challenge clips using OpenCV."""
        fps = 25
        duration_sec = 4  # 4 seconds each for crisp, punchy reaction rounds
        total_frames = fps * duration_sec
        width, height = 640, 360

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        themes = [
            ("JUMP SCARE FLASH", (0, 0, 255), "BOO!"),
            ("SPINNING SPIRAL", (255, 0, 128), "HYPNO FOCUS"),
            ("COUNTDOWN BLINK", (0, 255, 255), "STARE AT CENTER"),
            ("EXPANDING PULSE", (0, 255, 0), "DON'T BLINK!"),
            ("COLOR WARP", (255, 128, 0), "REACTION TEST"),
            ("SMILE TRAP", (200, 200, 0), "TRY NOT TO LAUGH"),
            ("SUDDEN GLITCH", (128, 0, 255), "SUSPENSE ALERT"),
            ("SPEED BOUNCE", (0, 128, 255), "TRACK THE BALL")
        ]

        for i in range(count):
            clip_idx = start_index + i
            out_file = self.video_dir / f"clip_{clip_idx:03d}.mp4"
            writer = cv2.VideoWriter(str(out_file), fourcc, fps, (width, height))

            theme_name, color, prompt = themes[clip_idx % len(themes)]
            bg_base = (int(color[0] * 0.2), int(color[1] * 0.2), int(color[2] * 0.2))

            for frame_idx in range(total_frames):
                frame = np.full((height, width, 3), bg_base, dtype=np.uint8)
                t = frame_idx / fps

                # Header info
                cv2.putText(frame, f"CHALLENGE #{clip_idx:02d} / 40", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(frame, prompt, (width // 2 - 140, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)

                # Procedural visual effect
                if "JUMP SCARE" in theme_name:
                    if frame_idx > total_frames - 20:  # sudden flash in final second
                        frame[:] = (0, 0, 255) if (frame_idx % 4 < 2) else (255, 255, 255)
                        cv2.putText(frame, "!!! GOTCHA !!!", (width // 2 - 180, height // 2),
                                    cv2.FONT_HERSHEY_DUPLEX, 1.4, (0, 0, 0), 4, cv2.LINE_AA)
                    else:
                        radius = max(5, int(35 + 25 * math.sin(t * 4)))
                        cv2.circle(frame, (width // 2, height // 2), radius, color, -1)

                elif "SPINNING" in theme_name:
                    angle = t * 360 * 2
                    cx, cy = width // 2, height // 2
                    for r in range(10, 120, 15):
                        offset_angle = math.radians(angle + r * 3)
                        x = int(cx + r * math.cos(offset_angle))
                        y = int(cy + r * math.sin(offset_angle))
                        cv2.circle(frame, (x, y), 8, color, -1)

                elif "COUNTDOWN" in theme_name:
                    remain = max(1, duration_sec - int(t))
                    cv2.putText(frame, str(remain), (width // 2 - 35, height // 2 + 40),
                                cv2.FONT_HERSHEY_DUPLEX, 3.0, (255, 255, 255), 6, cv2.LINE_AA)
                    cd_radius = max(5, int(70 + (t % 1.0) * 30))
                    cv2.circle(frame, (width // 2, height // 2), cd_radius, color, 3)

                elif "SPEED BOUNCE" in theme_name:
                    bx = int((width // 2) + (width // 3) * math.sin(t * 8))
                    by = int((height // 2) + (height // 4) * math.cos(t * 6))
                    cv2.circle(frame, (bx, by), 28, color, -1)
                    cv2.circle(frame, (bx, by), 32, (255, 255, 255), 2)

                else:
                    pulse = int(50 + 40 * math.sin(t * 6))
                    cv2.rectangle(frame, (width // 2 - pulse, height // 2 - pulse),
                                  (width // 2 + pulse, height // 2 + pulse), color, 4)

                # Progress bar at bottom
                progress = frame_idx / total_frames
                cv2.rectangle(frame, (20, height - 30), (int(20 + (width - 40) * progress), height - 20),
                              (0, 255, 0), -1)

                writer.write(frame)

            writer.release()
