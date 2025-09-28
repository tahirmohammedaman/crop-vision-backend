import os
from datetime import datetime
from uuid import uuid4
from fastapi import UploadFile


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def save_upload(media_root: str, file: UploadFile) -> str:
    today = datetime.utcnow()
    subdir = os.path.join(
        "images", f"{today.year}", f"{today.month:02}", f"{today.day:02}"
    )
    abs_dir = os.path.join(media_root, subdir)
    ensure_dir(abs_dir)
    ext = os.path.splitext(file.filename or "")[1].lower() or ".jpg"
    filename = f"{uuid4().hex}{ext}"
    abs_path = os.path.join(abs_dir, filename)
    with open(abs_path, "wb") as out:
        out.write(file.file.read())
    rel_path = os.path.join(subdir, filename).replace("\\", "/")
    return rel_path
