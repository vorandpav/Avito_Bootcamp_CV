import argparse
import random
import shutil
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image
from trdg.generators import GeneratorFromStrings

DICT_DIR = Path("data/raw/dictionaries")
FONTS_DIR = Path("assets/fonts")
BG_DIR = Path("assets/backgrounds")

RU_DICT_URL = "https://raw.githubusercontent.com/hingston/russian/master/100000-russian-words.txt"
EN_DICT_URL = "https://raw.githubusercontent.com/dwyl/english-words/master/words_alpha.txt"


def load_fonts() -> list[str]:
    """Шрифты с кириллицей из assets/fonts."""
    fonts = sorted(str(p) for p in FONTS_DIR.glob("*.ttf"))
    if not fonts:
        raise FileNotFoundError(f"Нет .ttf в {FONTS_DIR}. Положи туда Noto Sans (или другой шрифт с кириллицей).")
    return fonts


def ensure_dictionaries_loaded() -> tuple[list[str], list[str]]:
    """Скачивает полнотекстовые словари русского и английского языков."""
    DICT_DIR.mkdir(parents=True, exist_ok=True)
    ru_path = DICT_DIR / "russian_words.txt"
    en_path = DICT_DIR / "english_words.txt"

    if not ru_path.exists():
        urllib.request.urlretrieve(RU_DICT_URL, ru_path)

    if not en_path.exists():
        urllib.request.urlretrieve(EN_DICT_URL, en_path)

    with open(ru_path, "r", encoding="utf-8-sig", errors="ignore") as f:
        ru_words = [line.strip() for line in f if len(line.strip()) > 1]

    with open(en_path, "r", encoding="utf-8", errors="ignore") as f:
        en_words = [line.strip() for line in f if len(line.strip()) > 1]

    print(f"Загружено слов: {len(ru_words)} (RU), {len(en_words)} (EN).")

    return ru_words, en_words


def generate_random_text(ru_words: list[str], en_words: list[str]) -> str:
    """Генерирует случайный текст (1 слово, словосочетание, цифры, даты, телефоны)."""
    kind = random.choices(["ru", "en", "digits", "mixed"], weights=[0.55, 0.25, 0.10, 0.10])[0]

    if kind == "ru":
        num_words = random.choices([1, 2, 3], weights=[0.60, 0.30, 0.10])[0]
        return " ".join(random.sample(ru_words, num_words))
    elif kind == "en":
        num_words = random.choices([1, 2, 3], weights=[0.60, 0.30, 0.10])[0]
        return " ".join(random.sample(en_words, num_words))
    elif kind == "digits":
        digit_type = random.choice(["price", "phone", "date"])
        if digit_type == "price":
            return f"{random.randint(10, 99999)} руб"
        elif digit_type == "phone":
            return f"+7-{random.randint(900, 999)}-{random.randint(100, 999)}-{random.randint(10, 99)}"
        else:
            return f"{random.randint(1, 28):02d}.{random.randint(1, 12):02d}.2026"
    else:
        w_ru = random.choice(ru_words)
        w_en = random.choice(en_words)
        return f"{w_ru} {w_en}"


def _solid_bg(height: int, width: int) -> Image.Image:
    color = tuple(int(c) for c in np.random.randint(0, 256, size=3))
    return Image.new("RGB", (width, height), color)


def _noise_bg(height: int, width: int) -> Image.Image:
    mean = np.random.randint(40, 220)
    std = np.random.randint(8, 45)
    arr = np.clip(np.random.normal(mean, std, (height, width, 3)), 0, 255).astype(np.uint8)
    shift = np.random.randint(-25, 26, size=3)
    arr = np.clip(arr.astype(np.int16) + shift, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def _gradient_bg(height: int, width: int) -> Image.Image:
    c1 = np.random.randint(0, 256, size=3)
    c2 = np.random.randint(0, 256, size=3)
    if random.random() < 0.5:
        t = np.linspace(0, 1, width, dtype=np.float32)[None, :, None]
        arr = (c1 * (1 - t) + c2 * t).astype(np.uint8)
        arr = np.repeat(arr, height, axis=0)
    else:
        t = np.linspace(0, 1, height, dtype=np.float32)[:, None, None]
        arr = (c1 * (1 - t) + c2 * t).astype(np.uint8)
        arr = np.repeat(arr, width, axis=1)
    return Image.fromarray(arr, mode="RGB")


def ensure_background_pool(pool_size: int = 256, size: tuple[int, int] = (640, 320)) -> Path:
    """Пул solid/noise/gradient фонов — аугментацией такое разнообразие не получить."""
    BG_DIR.mkdir(parents=True, exist_ok=True)
    existing = list(BG_DIR.glob("*.png"))
    if len(existing) >= pool_size:
        return BG_DIR

    width, height = size
    makers = [_solid_bg, _noise_bg, _gradient_bg]
    for i in range(pool_size):
        out = BG_DIR / f"bg_{i:04d}.png"
        if out.exists():
            continue
        makers[i % len(makers)](height, width).save(out)

    print(f"Фонов в пуле: {len(list(BG_DIR.glob('*.png')))} ({BG_DIR})")
    return BG_DIR


def generate_synthetic_dataset(num_samples: int = 20000, output_dir: Path | None = None) -> None:
    """Генерирует синтетику через TRDG; фон — случайный кроп из пула картинок."""
    ru_words, en_words = ensure_dictionaries_loaded()

    output_dir = output_dir or Path("data/processed/synthetic_crops")
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Генерация {num_samples} кропов.")

    strings_list = [generate_random_text(ru_words, en_words) for _ in range(int(num_samples * 1.15))]

    fonts = load_fonts()
    random.shuffle(fonts)
    bg_dir = ensure_background_pool()
    print(f"Шрифтов: {len(fonts)}; фонов: {len(list(bg_dir.glob('*.png')))}")

    generator = GeneratorFromStrings(
        strings=strings_list,
        fonts=fonts,
        blur=1,
        random_blur=True,
        orientation=0,
        background_type=3,
        image_dir=str(bg_dir),
        size=32,
    )

    count = 0
    for img, _ in generator:
        if img is None:
            continue
        out_path = output_dir / f"crop_synth_{count:06d}.png"
        img.save(out_path)
        count += 1
        if count >= num_samples:
            break
        if count % 2000 == 0:
            print(f"  {count}/{num_samples}")

    print(f"Синтетика успешно сгенерирована ({count} шт.) в: {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic text dataset.")
    parser.add_argument("--samples", type=int, default=20000, help="Number of samples to generate")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generate_synthetic_dataset(num_samples=args.samples)


if __name__ == "__main__":
    main()
