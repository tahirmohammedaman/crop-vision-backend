import os
import random
import time
import requests
from glob import glob
from datetime import datetime

BASE_URL = "http://localhost:8000"
USERNAME = "admin"
PASSWORD = "admin"
COUNT = 15
PREDICT_PATH = "/v1/predict"
IMAGE_FIELD = "file"

DATASET_DIR = os.path.join(os.path.dirname(__file__), "data", "test-dataset")
EXTRA_TAGS_POOL = [
    "macro","field","leaf","mobile","lab","green","sunny","low-light","shadow",
    "water-drops","backlit","close-up","disease","healthy","stem","fruit","multi-leaf","high-res","no-flash","flash"
]
CROP_HINTS = ["full","center","top-left","top-right","bottom-left","bottom-right"]

def get_access_token(base_url, username, password):
    login_url = f"{base_url}/auth/login"
    data = {
        "username": username,
        "password": password,
        "scope": ""
    }
    resp = requests.post(login_url, data=data)
    resp.raise_for_status()
    token = resp.json().get("access_token")
    if not token:
        raise Exception("No access_token in response")
    return token

def get_tags_for_file(filename):
    base = filename.lower()
    tags = []
    if "apple" in base: tags.append("apple")
    if "potato" in base: tags.append("potato")
    if "tomato" in base: tags.append("tomato")
    if "corn" in base: tags.append("corn")
    if "rust" in base: tags.append("rust")
    if "scab" in base: tags.append("scab")
    if "earlyblight" in base: tags.append("early-blight")
    if "healthy" in base: tags.append("healthy")
    if "yellowcurlvirus" in base: tags.append("yellow-curl-virus")
    tags += random.sample(EXTRA_TAGS_POOL, random.randint(1, 3))
    return list(set(tags))

def main():
    if not os.path.isdir(DATASET_DIR):
        raise Exception(f"Dataset folder not found: {DATASET_DIR}")

    images = []
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        images += glob(os.path.join(DATASET_DIR, ext))
    if not images:
        raise Exception(f"No images found in {DATASET_DIR}")

    selected = random.sample(images, min(COUNT, len(images)))
    print(f"Logging in to {BASE_URL} ...")
    token = get_access_token(BASE_URL, USERNAME, PASSWORD)
    print("Got access token.")

    headers = {"Authorization": f"Bearer {token}"}
    predict_url = f"{BASE_URL}{PREDICT_PATH}"
    results = []

    for i, img_path in enumerate(selected, 1):
        tags = get_tags_for_file(os.path.basename(img_path))
        crop = random.choice(CROP_HINTS)
        files = {IMAGE_FIELD: open(img_path, "rb")}
        data = [("tags", t) for t in tags] + [("crop", crop)]
        print(f"[{i}/{len(selected)}] Posting {os.path.basename(img_path)} with tags: {', '.join(tags)} and crop: {crop}")
        try:
            resp = requests.post(predict_url, headers=headers, files=files, data=data)
            files[IMAGE_FIELD].close()
            if resp.status_code != 200:
                print(f"  Prediction failed ({resp.status_code}): {resp.text}")
                continue
            res = resp.json()
            results.append({
                "file": os.path.basename(img_path),
                "tags": ", ".join(tags),
                "crop": crop,
                "predicted_class": res.get("predicted_class"),
                "confidence": res.get("confidence"),
                "id": res.get("id"),
                "image_url": res.get("image_url"),
            })
            time.sleep(0.25)
        except Exception as e:
            print(f"  Error: {e}")

    print(f"\nDone. Created {len(results)} predictions.")
    for r in results:
        print(f"{r['file']:30} {r['predicted_class']:20} {r['confidence']:.3f} {r['tags']}")

    out_file = os.path.join(os.path.dirname(__file__), f"predictions_seed_{datetime.now():%Y%m%d_%H%M%S}.json")
    import json
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved summary to {out_file}")

if __name__ == "__main__":
    main()