import json
import zipfile
from pathlib import Path
import cv2
import pandas as pd
from tqdm import tqdm

DATA_DIR = Path('/kaggle/input/datasets/hardtype/rustitw-russian-language-visual-text-recognition/train/real')
OUTPUT_DIR = Path('/kaggle/working/rustitw_crops')
OUTPUT_ZIP = Path('/kaggle/working/rustitw_crops_1gb.zip')

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_DIR / 'info.csv')

crop_idx = 0
TARGET_MAX_CROPS = 35000

for _, row in tqdm(df.iterrows(), total=len(df)):
    if crop_idx >= TARGET_MAX_CROPS:
        break

    img_path = DATA_DIR / row['image_path']
    img = cv2.imread(str(img_path))

    img_h, img_w = img.shape[:2]
    bboxes = json.loads(row['box_and_label'])[0]

    for bbox in bboxes:
        x = int(bbox['left'] * img_w)
        y = int(bbox['top'] * img_h)
        w = int(bbox['width'] * img_w)
        h = int(bbox['height'] * img_h)

        if w < 8 or h < 8:
            continue

        crop = img[max(0, y): y + h, max(0, x): x + w]
        if crop.size == 0:
            continue

        cv2.imwrite(str(OUTPUT_DIR / f"crop_{crop_idx:06d}.png"), crop)
        crop_idx += 1

        if crop_idx >= TARGET_MAX_CROPS:
            break

with zipfile.ZipFile(OUTPUT_ZIP, 'w', zipfile.ZIP_DEFLATED) as zipf:
    for crop_file in tqdm(list(OUTPUT_DIR.glob("*.png"))):
        zipf.write(crop_file, arcname=crop_file.name)
