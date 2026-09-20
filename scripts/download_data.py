import argparse
import gzip
import json
from pathlib import Path
import tarfile
import urllib.request
import zipfile
import cv2
import gdown
import numpy as np
from tqdm import tqdm

from src.config import DataConfig, load_config

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def setup_rustitw(data_cfg: DataConfig) -> None:
    """Загружает и распаковывает RusTITW кропы с Google Drive."""
    target_dir = PROJECT_ROOT / data_cfg.processed_dir / "rustitw_crops"
    if target_dir.exists() and any(target_dir.iterdir()):
        print("RusTITW кропы уже распакованы.")
        return

    raw_dir = PROJECT_ROOT / data_cfg.raw_dir
    raw_dir.mkdir(parents=True, exist_ok=True)
    zip_path = raw_dir / "rustitw_crops_1gb.zip"

    if not zip_path.exists():
        print("Скачиваем RusTITW архив.")
        url = f"https://drive.google.com/uc?id={data_cfg.rustitw_gdrive_id}"
        gdown.download(url, str(zip_path), quiet=False)

    print("Распаковка RusTITW.")
    target_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(target_dir)

    print(f"RusTITW готов: {target_dir}")


def _download_with_progress(url: str, out_path: Path, desc: str) -> None:
    """Загрузка файла с индикацией tqdm."""
    part_path = out_path.with_suffix(".part")
    with urllib.request.urlopen(url, timeout=120) as resp:
        total_size = int(resp.headers.get("Content-Length", 0))
        block_size = 1024 * 1024

        with open(part_path, "wb") as out, tqdm(
                desc=desc,
                total=total_size,
                unit="iB",
                unit_scale=True,
                unit_divisor=1024,
        ) as pbar:
            while True:
                buffer = resp.read(block_size)
                if not buffer:
                    break
                out.write(buffer)
                pbar.update(len(buffer))

    part_path.replace(out_path)


def download_and_process_hiertext(data_cfg: DataConfig) -> None:
    """Загружает сырой HierText и нарезает текстовые кропы."""
    crop_dir = PROJECT_ROOT / data_cfg.processed_dir / "hiertext_crops"
    if crop_dir.exists() and any(crop_dir.iterdir()):
        print("HierText уже подготовлен.")
        return

    raw_dir = PROJECT_ROOT / data_cfg.raw_dir / "hiertext"
    raw_dir.mkdir(parents=True, exist_ok=True)

    for url, filename in [
        (data_cfg.hiertext_annotation_url, "validation.jsonl.gz"),
        (data_cfg.hiertext_archive_url, "validation.tgz"),
    ]:
        out_path = raw_dir / filename
        if not out_path.exists():
            _download_with_progress(url, out_path, desc=f"HierText [{filename}]")

    jsonl_path = raw_dir / "validation.jsonl.gz"
    archive_path = raw_dir / "validation.tgz"
    crop_dir.mkdir(parents=True, exist_ok=True)

    # HierText gt — один JSON-объект (не JSONL): {info, annotations:[{image_id, paragraphs:[{lines:[...]}]}]}
    with gzip.open(jsonl_path, "rt", encoding="utf-8") as f:
        payload = json.load(f)
    annotations = {item["image_id"]: item for item in payload["annotations"]}

    print("Обработка HierText.")
    crop_idx = 0

    with tarfile.open(archive_path, "r:gz") as tar:
        for member in tqdm(tar.getmembers()):
            if crop_idx >= data_cfg.hiertext_target_max_crops:
                break
            if not member.isfile() or not member.name.endswith((".jpg", ".png")):
                continue

            image_id = Path(member.name).stem
            if image_id not in annotations:
                continue

            file_obj = tar.extractfile(member)
            if file_obj is None:
                continue

            img_bytes = np.frombuffer(file_obj.read(), np.uint8)
            img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
            if img is None:
                continue

            img_info = annotations[image_id]
            for paragraph in img_info.get("paragraphs", []):
                for line_info in paragraph.get("lines", []):
                    vertices = np.array(line_info["vertices"], dtype=np.int32)
                    x, y, w, h = cv2.boundingRect(vertices)

                    if w < 8 or h < 8:
                        continue

                    crop = img[max(0, y): y + h, max(0, x): x + w]
                    if crop.size == 0:
                        continue

                    cv2.imwrite(str(crop_dir / f"crop_{crop_idx:06d}.png"), crop)
                    crop_idx += 1

                    if crop_idx >= data_cfg.hiertext_target_max_crops:
                        break
                if crop_idx >= data_cfg.hiertext_target_max_crops:
                    break

    print(f"Готово: нарезано {crop_idx} кропов HierText.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and prepare orientation datasets.")
    parser.add_argument("--config", type=Path, default=Path("configs/baseline.yaml"), help="Path to config file")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    setup_rustitw(config.data)

    download_and_process_hiertext(config.data)


if __name__ == "__main__":
    main()
