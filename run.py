from __future__ import annotations
import os
import sys
import webbrowser
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

# Safe UTF-8 encoding for Windows console
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from backend.config import load_config
from backend.video_manager import VideoManager
from backend.adb_manager import ADBManager

def main():
    print("=" * 65)
    print("   🎯 6-PLAYER LOCAL AI REACTION CHALLENGE SYSTEM")
    print("=" * 65)

    config = load_config()

    # 1. Ensure Challenge Videos Pool (40 videos)
    print("\n[1/3] Checking challenge video pool (40 clips)...")
    video_dir = PROJECT_ROOT / "data" / "videos"
    video_mgr = VideoManager(video_dir=video_dir, target_count=40)
    videos = video_mgr.refresh_and_ensure_videos()
    print(f"      ✓ {len(videos)} challenge videos ready in {video_dir}")

    # 2. Check ADB & Phone Status
    print("\n[2/3] Setting up pure USB Cable port forwarding for Phones...")
    adb_mgr = ADBManager()
    if adb_mgr.adb_bin:
        print(f"      ✓ ADB binary found at: {adb_mgr.adb_bin}")
        fw_res = adb_mgr.setup_port_forwarding(8081, 8082)
        print(f"      ✓ Active USB phone devices: {fw_res.get('authorized_devices', 0)}")
        for f in fw_res.get("forwards", []):
            print(f"         - Phone {f['phone_index']} ({f['serial']}) => {f['stream_url']} (100% USB)")
    else:
        print("      ℹ ADB not installed locally yet. You can auto-install via Dashboard!")

    # 3. Launch Server
    port = 8000
    host = "127.0.0.1"
    host_url = f"http://{host}:{port}/host"
    tv_url = f"http://{host}:{port}/tv"

    print("\n[3/3] Starting Local Server...")
    print(f"      📺 TV Display Window (Move to TV / Second Screen): {tv_url}")
    print(f"      🎛 Host / Referee Dashboard (Laptop Screen):      {host_url}")
    print("=" * 65)

    # Open host dashboard in browser
    try:
        webbrowser.open(host_url)
    except Exception:
        pass

    import uvicorn
    uvicorn.run("backend.app:app", host=host, port=port, log_level="info", reload=False)

if __name__ == "__main__":
    venv_python = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
    if venv_python.exists() and Path(sys.executable).resolve() != venv_python.resolve():
        import subprocess
        sys.exit(subprocess.call([str(venv_python), str(Path(__file__).resolve())] + sys.argv[1:]))
    main()
