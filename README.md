# Avito Bootcamp CV — ориентация текста

Классификация текстовых кропов по ориентации: **0° vs 180°**.

## Быстрый старт

```powershell
uv sync
$env:PYTHONPATH="."
uv run python scripts/check_config.py
uv run python scripts/generate_data.py --samples 20000
uv run python src/train.py
```

Подробное описание данных находится в [`data/README.md`](data/README.md).

## Данные

Пайплайн использует три возможных источника:

- **Synthetic** — 20 000 кропов.
- **RusTITW** — 35 000 кропов.
- **HierText** — 35 000 кропов.

При наличии всех источников используется до 90 000 исходных кропов.

Сначала исходные изображения разделяются на train / validation / internal test:

После split каждый исходный кроп превращается в два логических примера:

```text
original image -------------> label 0, rotate=False
same image rotated by 180° -> label 1, rotate=True
```

Поворот выполняется лениво в `Dataset.__getitem__`, поэтому отдельные копии изображений не создаются.

Это предотвращает leakage: оригинал и его 180°-версия всегда находятся в одном split.

По умолчанию:

```yaml
split:
  train_ratio: 0.8
  validation_ratio: 0.1
  test_ratio: 0.1
```

При 90 000 исходных кропов это примерно:

```text
72 000 source images -> train
 9 000 source images -> validation
 9 000 source images -> internal test
```

После создания двух вариантов:

```text
144 000 samples -> train
 18 000 samples -> validation
 18 000 samples -> internal test
```

`internal test` — размеченная отложенная выборка проекта. Она не используется для обучения или выбора checkpoint.

## Preprocessing

Baseline использует:

1. загрузку изображения;
2. преобразование в RGB;
3. resize до `96 × 384`;
4. `ToTensor`;
5. ImageNet normalization:

```text
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]
```

Дополнительных аугментаций в baseline пока нет.

## Baseline model

Текущая baseline-модель — **MobileNetV3-Large** с предобученными весами ImageNet.

Это готовая компактная CNN-архитектура из `torchvision`, а не самописная сеть.

Упрощённая схема:

```text
RGB image
   |
   v
Resize 96x384
   |
   v
MobileNetV3-Large backbone
   |
   v
classifier
   |
   +-- Dropout(0.2)
   |
   +-- Linear(..., 2)
           |
           +-- class 0 -> 0°
           +-- class 1 -> 180°
```

Модель выдаёт два логита, а обучение выполняется через `CrossEntropyLoss`.

MobileNetV3-Large выбрана как baseline: она достаточно компактная, быстро обучается на RTX 4060 и позволяет использовать
transfer learning.

Это именно **baseline**, а не финальная архитектура.

## Training

Обучение состоит из двух этапов.

### Stage 1 — frozen backbone

Backbone MobileNetV3 заморожен, обучается только классификационная голова.

```yaml
frozen_epochs: 1
frozen_learning_rate: 0.001
```

### Stage 2 — fine-tuning

Backbone размораживается и обучается вместе с classifier.

```yaml
finetune_epochs: 5
finetune_learning_rate: 0.0001
```

Оптимизатор:

```text
AdamW
weight_decay = 0.01
```

Batch size:

```text
32
```

Лучший checkpoint выбирается по `validation accuracy`.

## Baseline result

Первый frozen epoch дал:

```text
train_loss = 0.6318
train_acc  = 0.5930

val_loss   = 0.5818
val_acc    = 0.6244
```

То есть после первого этапа:

```text
Validation accuracy = 62.44%
```

Это промежуточный результат. После него должен выполняться fine-tuning.

## Train / validation / test

Роли наборов:

```text
train
  |
  +--> обучение параметров модели

validation
  |
  +--> выбор лучшего checkpoint

internal test
  |
  +--> финальная оценка после окончания обучения
```

`internal test` не должен участвовать в цикле обучения.

После завершения всех frozen + fine-tuning эпох нужно:

1. загрузить лучший `checkpoints/best.pt`;
2. создать `test_loader`;
3. прогнать лучшую модель по internal test;
4. посчитать `test_loss` и `test_accuracy`;
5. вывести финальный результат.

Это будет отдельный финальный прогон, а не часть обучения.

## Configuration

Сейчас `train.py` запускается без аргумента:

```powershell
python src/train.py
```

потому что `load_config()` по умолчанию использует:

```text
configs/baseline.yaml
```

Для текущего baseline это удобно.

Но по мере развития проекта лучше перейти на:

```powershell
python src/train.py --config configs/baseline.yaml
```

Тогда можно будет заводить отдельные эксперименты:

```text
configs/
    baseline.yaml
    mobilenet_finetune.yaml
    strong_augmentation.yaml
    ...
```

и менять параметры без изменения Python-кода.

Это особенно важно для следующих экспериментов: preprocessing, augmentations, architecture, learning rate, scheduler и
количество эпох можно будет сравнивать как отдельные конфигурации.

## Project structure

```text
Avito_Bootcamp_CV/
|
+-- configs/
|   +-- baseline.yaml
|
+-- data/
|   +-- README.md
|   +-- raw/
|   +-- processed/
|
+-- scripts/
|   +-- generate_data.py
|   +-- download_data.py
|   +-- check_config.py
|   +-- kaggle_crop_rustitw.py
|
+-- src/
|   +-- config.py
|   +-- reproducibility.py
|   +-- model.py
|   +-- trainer.py
|   +-- train.py
|   +-- data_preprocessing/
|
+-- checkpoints/
|
+-- pyproject.toml
+-- README.md
```

## План развития

Baseline является точкой отсчёта. Дальше отдельными экспериментами можно менять:

```text
Baseline
   |
   +-- preprocessing
   |     +-- resize/pad
   |     +-- augmentations
   |
   +-- model
   |     +-- другие CNN
   |     +-- более крупные модели
   |
   +-- training
   |     +-- learning rate
   |     +-- scheduler
   |     +-- batch size
   |     +-- epochs
   |
   +-- inference
         +-- TTA
         +-- threshold / calibration
```

Каждое изменение следует сравнивать с baseline на одном и том же internal test.

## Reproducibility

Seed задаётся в конфигурации:

```yaml
experiment:
  seed: 42
```

Перед созданием dataset и обучением вызывается:

```python
seed_everything(config.experiment.seed)
```

Устройство выбирается автоматически:

```text
CUDA available -> cuda
otherwise      -> cpu
```

На текущей машине baseline запускается на:

```text
Device: cuda
```
