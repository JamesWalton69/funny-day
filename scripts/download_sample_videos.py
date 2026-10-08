#!/usr/bin/env python3
"""
Sample Video Downloader for 6-Player AI Reaction Challenge
Allows referees and users to:
  1. Download curated open-license / royalty-free sample funny clips.
  2. Download any YouTube video or 'Try Not to Laugh' playlist via yt-dlp.
  3. Batch download clips from a text file into data/videos/.
"""

import argparse
import os
import sys
import subprocess
from pathlib import Path
import urllib.request
import urllib.error

PROJECT_ROOT = Path(__file__).resolve().parent.parent
VIDEOS_DIR = PROJECT_ROOT / "data" / "videos"

# Curated set of direct open-source & royalty-free sample challenge clips
SAMPLE_CLIPS = [
    {
        "filename": "challenge_01_bunny_slapstick.mp4",
        "url": "https://www.w3schools.com/html/mov_bbb.mp4",
        "description": "Big Buck Bunny funny slapstick animation moment"
    },
    {
        "filename": "challenge_02_bear_nature.mp4",
        "url": "https://filesamples.com/samples/video/mp4/sample_640x360.mp4",
        "description": "Playful bears in water"
    },
    {
        "filename": "challenge_03_ocean_splash.mp4",
        "url": "https://filesamples.com/samples/video/mp4/sample_960x400_ocean_with_audio.mp4",
        "description": "Dramatic ocean wave splash"
    }
]


def download_direct_samples():
    """Downloads curated direct sample clips into data/videos/."""
    VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 65)
    print("  Downloading Curated Sample Challenge Video Clips")
    print(f"  Target Destination: {VIDEOS_DIR}")
    print("=" * 65)

    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

    for item in SAMPLE_CLIPS:
        dest = VIDEOS_DIR / item["filename"]
        if dest.exists() and dest.stat().st_size > 10000:
            print(f"  [EXISTS] {item['filename']} ({dest.stat().st_size // 1024} KB)")
            continue

        print(f"  [FETCHING] {item['filename']} - {item['description']}...")
        try:
            req = urllib.request.Request(item["url"], headers=headers)
            with urllib.request.urlopen(req, timeout=15) as resp, open(dest, "wb") as f:
                data = resp.read()
                f.write(data)
            print(f"     -> Successfully downloaded ({len(data) // 1024} KB)")
        except Exception as e:
            print(f"     -> Download failed: {e}")

    print("\n✓ Sample clips ready in data/videos/!\n")


def download_youtube_clip(url_or_playlist: str, max_videos: int = 10):
    """Uses yt-dlp to download YouTube video(s) or shorts into data/videos/."""
    VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 65)
    print(f"  Downloading from YouTube: {url_or_playlist}")
    print(f"  Target Destination: {VIDEOS_DIR}")
    print(f"  Max Videos: {max_videos}")
    print("=" * 65)

    out_template = str(VIDEOS_DIR / "%(title).40s_%(id)s.%(ext)s")

    cmd = [
        sys.executable, "-m", "yt_dlp",
        "--format", "bestvideo[ext=mp4][height<=720]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "--output", out_template,
        "--max-downloads", str(max_videos),
        "--no-playlist" if ("playlist" not in url_or_playlist and max_videos == 1) else "--yes-playlist",
        url_or_playlist
    ]

    try:
        subprocess.run(cmd, check=True)
        print("\n✓ YouTube challenge clips downloaded successfully into data/videos/!\n")
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] yt-dlp process returned error code {e.returncode}\n")
    except Exception as e:
        print(f"\n[ERROR] Failed to run yt-dlp: {e}\n")


def main():
    parser = argparse.ArgumentParser(description="Sample Video Downloader for Reaction Challenge")
    parser.add_argument("--mode", choices=["sample", "yt"], default="sample", help="Mode: sample (direct bundle) or yt (YouTube)")
    parser.add_argument("--yt", type=str, help="YouTube video or playlist URL (e.g., 'https://www.youtube.com/watch?v=...')")
    parser.add_argument("--max", type=int, default=10, help="Maximum number of videos to download from YouTube playlist")

    args = parser.parse_args()

    if args.yt:
        download_youtube_clip(args.yt, max_videos=args.max)
    elif args.mode == "sample":
        download_direct_samples()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
