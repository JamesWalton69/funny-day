from __future__ import annotations
import os
import shutil
import subprocess
import urllib.request
import zipfile
from pathlib import Path
from typing import List, Dict, Any, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TOOLS_DIR = PROJECT_ROOT / "tools" / "platform-tools"
ADB_DOWNLOAD_URL = "https://dl.google.com/android/repository/platform-tools-latest-windows.zip"


class ADBManager:
    def __init__(self, tools_dir: Optional[Path] = None):
        self.tools_dir = tools_dir or DEFAULT_TOOLS_DIR
        self.adb_bin = self._find_or_prepare_adb()

    def _find_or_prepare_adb(self) -> Optional[str]:
        # 1. Check local tools directory
        local_adb = self.tools_dir / "adb.exe"
        if local_adb.exists():
            return str(local_adb)

        # 2. Check if adb is in system PATH
        system_adb = shutil.which("adb")
        if system_adb:
            return system_adb

        return None

    def ensure_adb_installed(self) -> bool:
        """Downloads portable Android platform-tools if adb is not found."""
        if self.adb_bin and Path(self.adb_bin).exists():
            return True

        local_adb = self.tools_dir / "adb.exe"
        if local_adb.exists():
            self.adb_bin = str(local_adb)
            return True

        tools_parent = self.tools_dir.parent
        tools_parent.mkdir(parents=True, exist_ok=True)
        zip_path = tools_parent / "platform-tools.zip"

        try:
            print(f"Downloading portable Android Platform-Tools from {ADB_DOWNLOAD_URL}...")
            urllib.request.urlretrieve(ADB_DOWNLOAD_URL, str(zip_path))
            print("Extracting platform-tools...")
            with zipfile.ZipFile(str(zip_path), "r") as zip_ref:
                zip_ref.extractall(str(tools_parent))

            if zip_path.exists():
                zip_path.unlink()

            if local_adb.exists():
                self.adb_bin = str(local_adb)
                print(f"ADB successfully installed at {self.adb_bin}")
                return True
        except Exception as e:
            print(f"Error downloading/extracting ADB: {e}")

        return False

    def run_adb(self, args: List[str]) -> subprocess.CompletedProcess:
        if not self.adb_bin:
            raise RuntimeError("ADB binary not found. Please run ensure_adb_installed() first.")
        cmd = [self.adb_bin] + args
        return subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=10,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        )

    def get_connected_devices(self) -> List[Dict[str, str]]:
        if not self.adb_bin:
            return []
        try:
            res = self.run_adb(["devices", "-l"])
            lines = res.stdout.strip().split("\n")
            devices = []
            for line in lines[1:]:  # skip 'List of devices attached'
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) >= 2:
                    serial = parts[0]
                    state = parts[1]
                    info = " ".join(parts[2:]) if len(parts) > 2 else ""
                    devices.append({
                        "serial": serial,
                        "state": state,
                        "info": info
                    })
            return devices
        except Exception as e:
            print(f"Error querying adb devices: {e}")
            return []

    def setup_port_forwarding(self, phone1_port: int = 8081, phone2_port: int = 8082, target_port: int = 8080) -> Dict[str, Any]:
        """Automatically forwards ports for connected USB phones."""
        if not self.adb_bin:
            installed = self.ensure_adb_installed()
            if not installed:
                return {
                    "success": False,
                    "error": "ADB is not available and download failed.",
                    "phones": []
                }

        devices = self.get_connected_devices()
        active_devices = [d for d in devices if d["state"] == "device"]

        result = {
            "success": True,
            "total_detected": len(devices),
            "authorized_devices": len(active_devices),
            "forwards": []
        }

        ports = [phone1_port, phone2_port]
        for idx, dev in enumerate(active_devices[:2]):
            local_port = ports[idx]
            serial = dev["serial"]
            try:
                # Run: adb -s <serial> forward tcp:<local_port> tcp:<target_port>
                self.run_adb(["-s", serial, "forward", f"tcp:{local_port}", f"tcp:{target_port}"])
                result["forwards"].append({
                    "phone_index": idx + 1,
                    "serial": serial,
                    "local_port": local_port,
                    "stream_url": f"http://127.0.0.1:{local_port}/video",
                    "status": "forwarded"
                })
            except Exception as e:
                result["forwards"].append({
                    "phone_index": idx + 1,
                    "serial": serial,
                    "local_port": local_port,
                    "error": str(e),
                    "status": "failed"
                })

        return result

    def get_status(self, phone1_port: int = 8081, phone2_port: int = 8082) -> Dict[str, Any]:
        has_adb = bool(self.adb_bin and Path(self.adb_bin).exists())
        devices = self.get_connected_devices() if has_adb else []
        forward_rules = []
        if has_adb:
            try:
                res = self.run_adb(["forward", "--list"])
                for line in res.stdout.strip().split("\n"):
                    if line.strip():
                        forward_rules.append(line.strip())
            except Exception:
                pass

        return {
            "adb_available": has_adb,
            "adb_path": self.adb_bin,
            "connected_devices": devices,
            "active_forwards": forward_rules,
            "configured_ports": {
                "phone1": phone1_port,
                "phone2": phone2_port
            }
        }
