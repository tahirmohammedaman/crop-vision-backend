#!/usr/bin/env python3
# filepath: ./device_predict.py
import argparse
import pathlib
import sys
from typing import List

import requests

API_URL = "https://your-api-host/v1/predict"
DEVICE_ID = "replace-with-device-id"
DEVICE_API_KEY = "replace-with-device-api-key"
TIMEOUT = 30


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Send prediction requests as a registered edge device."
    )
    parser.add_argument("image_path", type=pathlib.Path, help="Image file to upload")
    parser.add_argument(
        "tags",
        nargs="*",
        help="Optional tags (space-separated). Example: tag1 tag2 tag3",
    )
    return parser.parse_args(argv)


def send_prediction(image_path: pathlib.Path, tags: List[str]) -> None:
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    files = {"file": (image_path.name, image_path.open("rb"), "image/jpeg")}
    data = {
        "origin": "server_edge",
        "device_id": DEVICE_ID,
    }
    for tag in tags:
        data.setdefault("tags", []).append(tag)

    headers = {
        "X-Device-ID": DEVICE_ID,
        "X-Device-Key": DEVICE_API_KEY,
    }

    response = requests.post(
        API_URL,
        headers=headers,
        data=data,
        files=files,
        timeout=TIMEOUT,
    )
    response.raise_for_status()
    print("Prediction submitted successfully:")
    print(response.json())


def main(argv: List[str]) -> None:
    args = parse_args(argv)
    send_prediction(args.image_path, args.tags)


if __name__ == "__main__":
    try:
        main(sys.argv[1:])
    except Exception as exc:
        print(f"Prediction failed: {exc}", file=sys.stderr)
        sys.exit(1)
