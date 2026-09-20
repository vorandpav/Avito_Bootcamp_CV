# Avito Bootcamp CV - ориентация текстового кропа (0° / 180°)

Классификация OCR-кропа: нормальная ориентация vs перевёрнут вверх ногами. Модель выдаёт `p_180` ∈ [0, 1]. Метрика
конкурса: **1 − Brier Score**.

В проде такой классификатор крутится на каждом боксе каждого изображения (десятки-сотни миллионов вызовов в день),
поэтому важны не только качество, но и скорость и размер модели.

---

## Что сделано

| Блок                                                | Статус |
|-----------------------------------------------------|--------|
| Данные: синтетика (TRDG), RusTITW, HierText         | готово |
| Split без leakage (оригинал и 180° в одном split)   | готово |
| Train / val / internal test + checkpoint по 1−Brier | готово |
| Двухфазное обучение: freeze -> fine-tune            | готово |
| Аугментации под реальные OCR-кропы                  | готово |
| MobileNetV3-Large / Small, 96×384 и 128×512         | готово |
| Inference -> `submission.csv` (`image_id`, `p_180`) | готово |
| Логи прогонов рядом с конфигами (`configs/*.txt`)   | готово |
 / Бенчмарк скорости инференса                       | готово |

Open-source (бесплатная коммерческая лицензия): PyTorch / torchvision (MobileNetV3 + ImageNet weights), Pillow,
Albumentations-соседние кастомные
transforms, TRDG, scikit-learn, pandas, pydantic, kagglehub/gdown для данных.

---

## Быстрый старт

```powershell
uv sync
$env:PYTHONPATH="."

# проверка конфига и CUDA
uv run python scripts/check_config.py

# данные (синтетика; реальные)
uv run python scripts/generate_data.py --samples 20000
uv run python scripts/download_data.py --config configs/baseline.yaml

# обучение итоговой модели
uv run python src/train.py --config configs/mobilenet_small_128x512.yaml

# обучение всех экспериментов
uv run python src/train.py --config configs/baseline.yaml
uv run python src/train.py --config configs/augmentation_v1.yaml
uv run python src/train.py --config configs/mobilenet_small.yaml

# сабмит на тестовые кропы конкурса
uv run python src/inference.py --config configs/mobilenet_small_128x512.yaml --input-dir data/test/images --output submission.csv

# сравнение скорости инференса
uv run python scripts/benchmark_inference.py
```

Подробности по данным: [`data/README.md`](data/README.md).

Checkpoint сохраняется в `checkpoints/<experiment.name>/best.pt`. Inference сам подхватывает его по имени эксперимента
из YAML.

---

## Данные и лейблы

Источники (90k исходных кропов):

- **Synthetic** - 20k (обязательный путь для воспроизводимости)
- **RusTITW** - 35k (реальные RU/EN текстовые кропы)
- **HierText** - 35k (строки из Open Images / HierText)

Split по **исходным** картинкам (80 / 10 / 10), затем каждый кроп даёт два примера:

```text
original  -> label 0 (rotate=False)
same@180° -> label 1 (rotate=True)
```

Поворот ленивый в `Dataset.__getitem__` - дубликаты на диск не пишутся. Оригинал и 180° всегда в одном split -> нет
leakage.

---

## Пайплайн модели

```text
RGB crop
  -> Resize (H×W из конфига)
  -> [train only] аугментации
  -> ToTensor + ImageNet normalize
  -> MobileNetV3 backbone
  -> Dropout -> Linear(..., 2)
  -> softmax -> p_180 = P(class=1)
```

Обучение: `CrossEntropyLoss`, AdamW (`wd=0.01`), batch 32.

1. **Frozen** (2 эпохи, lr=1e-3) - учится только голова
2. **Fine-tune** (10 эпох, lr=1e-4) - весь backbone

---

## Конфиги

### 1. `baseline.yaml` - точка отсчёта

| Параметр    | Значение                    |
|-------------|-----------------------------|
| Модель      | MobileNetV3-Large, ImageNet |
| Вход        | **96×384**                  |
| Аугментации | выкл.                       |

**Зачем:** измерить transfer learning "как есть" на чистых кропах. Без аугментаций видно чистый ceiling по fit/val и
есть якорь для сравнения.

**Почему подходит кейсу:** задача простая (2 класса), кропы уже вырезаны детектором - тяжёлая ViT/LLM не нужна. Large
даёт запас качества при всё ещё мобильном backbone.

**Результат:** internal test acc = 0.91; лидерборд **1−Brier = 0.9474**.

### 2. `augmentation_v1.yaml` - закрытие domain gap

Тот же Large + 96×384, но train-ауги:

- лёгкий/средний поворот - кривые фото/сканы
- color jitter - освещение объявлений
- perspective - перспектива камеры
- blur / JPEG / gaussian noise - сжатие и качество загрузок

**Зачем:** тест конкурса и прод - грязные реальные боксы; чистый baseline переобучается на "идеальные" кропы (val acc
выше, LB хуже).

**Почему идеально для кейса:** ауги имитируют именно то, что видит OCR на Авито, без разрушения читаемости текста (нет
aggressive crop/erase).

**Результат:** val acc ниже baseline (0.88), но лидерборд выше - **1−Brier = 0.9557**. Главный качественный шаг.

### 3. `mobilenet_small.yaml` - вариант на скорость/размер

Тот же пайплайн и ауги, модель -> MobileNetV3-Small, вход 96×384.

**Зачем:** при близком скоре побеждает более быстрая/компактная модель. Checkpoint 6.2 MB vs 17 MB у Large.

**Почему ограничение разумное:** Small - естественный кандидат в прод; если качество не проседает критично, это и есть
целевой trade-off задачи.

**Результат:** LB **1−Brier = 0.9311** - заметный дроп. Small на 96×384 пока слабее Large+aug.

### 4. `mobilenet_small_128x512.yaml` - вернуть качество без смены семейства

Small + ауги, вход **128×512** (тот же aspect 1:4, больше пикселей на буквы).

**Зачем:** гипотеза - Small теряет на мелком шрифте/узких глифах; увеличение резолюции дешевле, чем переход на Large или
тяжёлые архитектуры.

**Результат:** гипотеза подтвердилась - LB **1−Brier = 0.9505** (internal test score 0.9088). Почти уровень Large+aug
при втрое меньшем checkpoint.

### Почему ограничился именно этим набором

```text
качество ──► baseline -> +aug -> small -> small@128×512
скорость ──► Large     Large   Small  Small (чуть медленнее 96×384)
```

1. Одна семья моделей (MobileNetV3) - fair comparison, один inference-код.
2. Aspect 1:4 зафиксирован под горизонтальные OCR-боксы - не квадрат ImageNet.
3. Не уходим в EfficientNet-L / ConvNeXt / ViT: на бинарной ориентации выигрыш сомнителен, latency - нет.
4. Аугментации - один осмысленный пакет под домен, а не 20 случайных yaml.

---

## Сводка результатов

| Конфиг                    | Модель   | Вход    | Aug | Val / Test (internal)             | LB 1−Brier |
|---------------------------|----------|---------|-----|-----------------------------------|------------|
| `baseline`                | V3-Large | 96×384  | нет | best val acc 0.915 / test 0.912   | 0.9474     |
| `augmentation_v1`         | V3-Large | 96×384  | да  | best val 0.880 / test 0.884       | **0.9557** |
| `mobilenet_small`         | V3-Small | 96×384  | да  | best val score 0.895 / test 0.894 | 0.9311     |
| `mobilenet_small_128x512` | V3-Small | 128×512 | да  | best val score 0.909 / test 0.909 | **0.9505** |

Полные логи эпох: `configs/*.txt`.

- Max quality: **`augmentation_v1`** (0.9557).
- Best speed/quality: **`mobilenet_small_128x512`** (0.9505 при 1.7x быстрее и 2.7x легче Large).

---

## Скорость инференса

Замер: CUDA, batch 32, 20 warmup / 100 timed runs, только forward (без I/O диск->GPU).

```powershell
uv run python scripts/benchmark_inference.py
```

| experiment                | input   | params | ckpt    | ms/img | img/s | LB 1−Brier |
|---------------------------|---------|--------|---------|--------|-------|------------|
| `augmentation_v1`         | 96×384  | 4.20M  | 16.2 MB | 0.668  | 1498  | **0.9557** |
| `mobilenet_small_128x512` | 128×512 | 1.52M  | 5.9 MB  | 0.398  | 2510  | **0.9505** |
| `mobilenet_small`         | 96×384  | 1.52M  | 5.9 MB  | 0.252  | 3972  | 0.9311     |
| `baseline`                | 96×384  | 4.20M  | 16.2 MB | 0.674  | 1483  | 0.9474     |

`mobilenet_small_128x512` отстаёт от лучшего скора на 0.005, но даёт **~1.7x throughput** и checkpoint **5.9 MB vs 16.2
MB**.

---

## Структура репозитория

```text
Avito_Bootcamp_CV/
├── configs/                 # yaml экспериментов + txt-логи
├── data/
│   ├── raw/                 # словари, архивы
│   ├── processed/           # synthetic / rustitw / hiertext crops
│   └── test/images/         # 20k тестовых кропов конкурса
├── scripts/
│   ├── generate_data.py
│   ├── download_data.py
│   ├── kaggle_crop_rustitw.py
│   ├── check_config.py
│   └── benchmark_inference.py
├── src/
│   ├── train.py             # обучение по --config
│   ├── inference.py         # submission.csv
│   ├── model.py             # MobileNetV3 Large/Small
│   ├── trainer.py
│   ├── config.py            # pydantic-схема YAML
│   ├── experiments/         # baseline trainer
│   └── data_preprocessing/  # dataset, split, transforms
├── checkpoints/<name>/best.pt
├── pyproject.toml
└── README.md
```

---

## Воспроизводимость

- Seed: `experiment.seed: 42` -> `seed_everything` до датасета и обучения.
- Конфиг - единственный источник правды (модель, размер входа, ауги, эпохи, lr).
- Устройство: CUDA если есть, иначе CPU.
- Зависимости зафиксированы через `uv.lock` (`uv sync`).

---

## Выводы

1. Чистый Large baseline уже сильный (~0.947), но переоценивает себя на чистом val относительно лидерборда.
2. Domain-аугментации - главный буст качества: val accuracy падает, LB растёт до **0.9557** (`augmentation_v1`).
3. Small@96×384 слишком слабый (0.931). Поднятие входа до 128×512 почти закрывает gap с Large (**0.9505**) без смены
   архитектуры.
4. Для прода рациональный выбор - **`mobilenet_small_128x512`**: почти тот же score, в ~1.7× быстрее и в ~2.7× легче
   Large. Если нужен максимум метрики - `augmentation_v1`.
5. Тяжёлые модели / LLM не нужны: бинарная ориентация текстового кропа хорошо решается компактным MobileNet + доменными
   аугами.
