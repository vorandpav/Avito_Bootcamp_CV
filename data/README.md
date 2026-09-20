# Данные (`/data`)

Пайплайн готовит кропы текста (BBox) для классификации ориентации **0° / 180°**.

## Анализ тестовой выборки

По визуальному разбору тестовых картинок конкурса:

| Наблюдение   | Детали                                                                            |
|--------------|-----------------------------------------------------------------------------------|
| Языки        | 70% русский (кириллица), 30% английский + цифры/спецсимволы; других алфавитов нет |
| Формат       | Уже нарезанные текстовые кропы, не полные сцены                                   |
| Геометрия    | Горизонтально вытянутые прямоугольники разного масштаба                           |
| Длина текста | 60% одно слово, 30% 2–3 слова, 10% длинные фразы                                  |

Базовый путь для проверки репозитория - синтетика. Реальные датасеты опциональны (если не лень качать 2.4 ГБ).

## Источники

1. **[Синтетика (TRDG)](https://github.com/Belval/TextRecognitionDataGenerator)** - обязательный путь,
   `scripts/generate_data.py`  
   Пакет `trdg` ставится через `uv sync`. Скрипт качает словари RU/EN и генерирует слова, цены, даты, телефоны под
   распределение выше. Результат: `data/processed/synthetic_crops/`.
2. **[RusTITW](https://www.kaggle.com/datasets/hardtype/rustitw-russian-language-visual-text-recognition)** -
   опционально, `scripts/download_data.py`  
   Архив кропов (1 ГБ, 35k) с Google Drive. Исходная нарезка: `scripts/kaggle_crop_rustitw.py` (Kaggle).

3. **[HierText](https://github.com/google-research-datasets/hiertext)** - опционально, `scripts/download_data.py`  
   Validation Open Images + аннотации Google HierText, нарезка строк в 35k кропов.

## Структура

```
data/
  raw/
    dictionaries/       
    rustitw_crops_1gb.zip 
    hiertext/            
  processed/
    synthetic_crops/       # после generate_data (базовый минимум)
    rustitw_crops/         # опционально
    hiertext_crops/        # опционально
```

Обучение (отдельный пайплайн) само подхватит доступные папки из `processed/`, смешает, сделает split и аугментации
0°/180°.

## Как запустить

```powershell
uv sync
$env:PYTHONPATH="."
```

### Рекомендуется проверяющему: только синтетика

```powershell
uv run python scripts/generate_data.py --samples 20000
```

### Опционально: реальные данные (RusTITW + HierText)

```powershell
uv run python scripts/download_data.py --config configs/baseline.yaml
```

Повторный запуск безопасен: готовые `processed/*_crops` пропускаются.

## Параметры

Лимиты и URL реальных датасетов: `configs/baseline.yaml` (`data.rustitw_*`, `data.hiertext_*`).
