import json
import os
import requests
from pathlib import Path
from tqdm import tqdm

# ================= CONFIG =================
NDJSON_PATH = "evo.ndjson"
OUTPUT_DIR = "dataset"
CLASS_NAMES = ["Evo"]  # ganti sesuai class kamu
# =========================================

IMG_DIR = Path(OUTPUT_DIR) / "images"
LBL_DIR = Path(OUTPUT_DIR) / "labels"

for split in ["train", "val"]:
    (IMG_DIR / split).mkdir(parents=True, exist_ok=True)
    (LBL_DIR / split).mkdir(parents=True, exist_ok=True)


def download_image(url, save_path):
    r = requests.get(url, timeout=10)
    r.raise_for_status()
    with open(save_path, "wb") as f:
        f.write(r.content)


with open(NDJSON_PATH, "r", encoding="utf-8") as f:
    lines = f.readlines()

for line in tqdm(lines, desc="Converting NDJSON"):
    data = json.loads(line)

    img_name = data["file"]
    img_url = data["url"]
    split = data.get("split", "train")

    img_path = IMG_DIR / split / img_name
    label_path = LBL_DIR / split / img_name.replace(Path(img_name).suffix, ".txt")

    # download image
    if not img_path.exists():
        download_image(img_url, img_path)

    # write label
    with open(label_path, "w") as lf:
        for box in data["annotations"]["boxes"]:
            cls, xc, yc, w, h = box
            lf.write(f"{cls} {xc} {yc} {w} {h}\n")


# create data.yaml
yaml_content = f"""
path: {Path(OUTPUT_DIR).absolute()}
train: images/train
val: images/val

names:
"""
for i, name in enumerate(CLASS_NAMES):
    yaml_content += f"  {i}: {name}\n"

with open(Path(OUTPUT_DIR) / "data.yaml", "w") as yf:
    yf.write(yaml_content)

print("✅ Conversion finished. Dataset ready for YOLO.")