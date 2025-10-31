#!/usr/bin/env python3
# filepath: ./device_heartbeat.py
import json
import logging
import shutil
import subprocess
import sys
import time
from datetime import datetime
from typing import Any

try:
    import psutil  # type: ignore
except ImportError:  # pragma: no cover
    print(
        "psutil is required. Install with: sudo apt-get install python3-psutil",
        file=sys.stderr,
    )
    sys.exit(1)

import requests

API_URL = "https://your-api-host/v1/devices/{device_id}/heartbeat"
DEVICE_ID = "replace-with-device-id"
DEVICE_API_KEY = "replace-with-device-api-key"

SEND_INTERVAL_SECONDS = 600  # 10 minutes
RETRY_ATTEMPTS = 3
RETRY_DELAY_SECONDS = 5

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


def _read_cpu_temp() -> float | None:
    temps = psutil.sensors_temperatures(fahrenheit=False)
    if temps:
        for entries in temps.values():
            for entry in entries:
                if entry.current is not None:
                    return entry.current
    try:
        with open("/sys/class/thermal/thermal_zone0/temp", "r", encoding="utf-8") as fh:
            return float(fh.read().strip()) / 1000.0
    except (FileNotFoundError, ValueError):
        return None


def _camera_status() -> dict[str, Any]:
    vcgencmd = shutil.which("vcgencmd")
    if vcgencmd:
        try:
            res = subprocess.run(
                [vcgencmd, "get_camera"],
                check=False,
                capture_output=True,
                text=True,
                timeout=5,
            )
            if res.returncode == 0:
                supported = "supported=1" in res.stdout
                detected = "detected=1" in res.stdout
                return {
                    "available": bool(supported and detected),
                    "detail": res.stdout.strip(),
                    "source": "vcgencmd",
                }
        except subprocess.SubprocessError as exc:
            logging.debug("vcgencmd check failed: %s", exc)

    libcamera = shutil.which("libcamera-hello")
    if libcamera:
        try:
            res = subprocess.run(
                [libcamera, "--list-cameras"],
                check=False,
                capture_output=True,
                text=True,
                timeout=15,
            )
            available = res.returncode == 0 and bool(res.stdout.strip())
            return {
                "available": available,
                "detail": res.stdout.strip() if available else res.stderr.strip(),
                "source": "libcamera-hello",
            }
        except subprocess.SubprocessError as exc:
            logging.debug("libcamera check failed: %s", exc)

    return {
        "available": False,
        "detail": "No camera detected (vcgencmd/libcamera unavailable).",
        "source": "fallback",
    }


def collect_metrics() -> dict[str, Any]:
    cpu_percent = psutil.cpu_percent(interval=0.2)
    mem = psutil.virtual_memory()
    disk = psutil.disk_usage("/")
    load1, load5, load15 = psutil.getloadavg()
    uptime_seconds = int(time.time() - psutil.boot_time())
    temperature = _read_cpu_temp()

    metrics: dict[str, Any] = {
        "cpu_percent": cpu_percent,
        "mem_percent": round(mem.percent, 2),
        "disk_percent": round(disk.percent, 2),
        "load_avg": {"1m": load1, "5m": load5, "15m": load15},
        "uptime_seconds": uptime_seconds,
        "camera": _camera_status(),
    }
    if temperature is not None:
        metrics["temp_c"] = round(temperature, 2)
    return metrics


def send_heartbeat(metrics: dict[str, Any]) -> None:
    url = API_URL.format(device_id=DEVICE_ID)
    payload = {"metrics": metrics}
    headers = {
        "Content-Type": "application/json",
        "X-Device-ID": DEVICE_ID,
        "X-Device-Key": DEVICE_API_KEY,
    }

    for attempt in range(1, RETRY_ATTEMPTS + 1):
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=10)
            response.raise_for_status()
            logging.info("Heartbeat accepted at %s", datetime.utcnow().isoformat())
            return
        except requests.RequestException as exc:
            logging.warning("Heartbeat attempt %d failed: %s", attempt, exc)
            if attempt < RETRY_ATTEMPTS:
                time.sleep(RETRY_DELAY_SECONDS)
    logging.error("Heartbeat failed after %d attempts", RETRY_ATTEMPTS)


def main() -> None:
    logging.info("Starting heartbeat loop (interval: %ss)", SEND_INTERVAL_SECONDS)
    while True:
        metrics = collect_metrics()
        logging.debug("Metrics payload: %s", json.dumps(metrics))
        send_heartbeat(metrics)
        time.sleep(SEND_INTERVAL_SECONDS)


if __name__ == "__main__":
    main()
