import os
import sys
from pathlib import Path
import yt_dlp
import imageio_ffmpeg

# Safe UTF-8 console output for Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
VIDEOS_DIR = PROJECT_ROOT / "data" / "videos"
PLAYLIST_URL = "https://youtube.com/playlist?list=PLPPomK5QKeyWV7PYC9s-PxrDhVIDpt4Oe"
TARGET_COUNT = 40

def sanitize_title(title: str) -> str:
    cleaned = "".join(c for c in title if c.isalnum() or c in (" ", "_", "-")).rstrip()
    return cleaned[:30].strip().replace(" ", "_")

def main():
    VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

    print("=" * 65)
    print("  Downloading 40 Videos from Challenge Playlist")
    print(f"  Target Destination: {VIDEOS_DIR}")
    print(f"  Playlist URL:       {PLAYLIST_URL}")
    print(f"  FFmpeg Binary:      {ffmpeg_exe}")
    print("=" * 65)

    ydl_opts_info = {
        "extract_flat": True,
        "quiet": True,
        "no_warnings": True
    }

    with yt_dlp.YoutubeDL(ydl_opts_info) as ydl:
        print("Fetching playlist metadata...")
        playlist_dict = ydl.extract_info(PLAYLIST_URL, download=False)

    entries = playlist_dict.get("entries", [])
    print(f"Found {len(entries)} items in playlist. Downloading first {TARGET_COUNT} valid clips...\n")

    success_count = 0
    downloaded_files = []

    for idx, entry in enumerate(entries, 1):
        if success_count >= TARGET_COUNT:
            break

        vid_id = entry.get("id")
        title = entry.get("title")

        if not vid_id or not isinstance(title, str) or title.strip() == "NA" or "[Private" in title or "[Deleted" in title:
            print(f"  [--] Skipping unavailable item: {title}")
            continue

        clean_name = sanitize_title(title)
        out_base = f"challenge_{success_count + 1:02d}_{clean_name}"
        vid_url = f"https://www.youtube.com/watch?v={vid_id}"

        print(f"  [{success_count + 1:02d}/{TARGET_COUNT:02d}] Downloading: {title} ({vid_id})...")

        ydl_opts_down = {
            "format": "bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/best[height<=720][ext=mp4]/best",
            "outtmpl": str(VIDEOS_DIR / f"{out_base}.%(ext)s"),
            "ffmpeg_location": ffmpeg_exe,
            "merge_output_format": "mp4",
            "quiet": True,
            "no_warnings": True
        }

        try:
            with yt_dlp.YoutubeDL(ydl_opts_down) as ydl_down:
                ydl_down.download([vid_url])

            matches = list(VIDEOS_DIR.glob(f"{out_base}.*"))
            if matches and matches[0].stat().st_size > 5000:
                final_file = matches[0]
                success_count += 1
                downloaded_files.append(final_file)
                print(f"        -> [OK] Saved as {final_file.name} ({final_file.stat().st_size // 1024} KB)")
            else:
                print(f"        -> [SKIP] 0-byte or failed output.")
        except Exception as e:
            print(f"        -> [ERROR] Could not download {vid_id}: {e}")

    print("\n" + "=" * 65)
    print(f"  ✓ Successfully downloaded {success_count} challenge videos into data/videos/!")
    print("=" * 65)

if __name__ == "__main__":
    main()
