"""Настройки приложения: хранятся в settings.json и меняются из веб-интерфейса."""
from __future__ import annotations

import json
import re
import threading
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path

# Каталог моделей. Скачиваются автоматически при первом выборе.
# Метрики — официальные данные Ultralytics при входе 640 px:
#   map  — mAP50-95 (%) на своём наборе данных (COCO, Open Images V7 или LVIS),
#   params — млн параметров, flops — млрд операций на кадр,
#   cpu_ms — скорость на CPU (ONNX), по данным Ultralytics (для YOLOE не публикуется).
SIZE_NAMES = {"n": "Nano", "s": "Small", "m": "Medium", "l": "Large", "x": "Extra Large"}
SIZE_HINTS = {
    "n": "самая быстрая, для любого процессора",
    "s": "баланс скорости и точности",
    "m": "точнее, лучше с видеокартой",
    "l": "высокая точность, нужна видеокарта",
    "x": "максимальная точность, мощная видеокарта",
}
FAMILIES = {
    # ключ:      (группа в списке,                        набор данных,        классов, произвольные классы)
    "yolo26":    ("YOLO26 — 80 классов COCO",               "COCO",              80,  False),
    "yolo11":    ("YOLO11 — 80 классов COCO",               "COCO",              80,  False),
    "yolov8":    ("YOLOv8 — 80 классов COCO",               "COCO",              80,  False),
    "oiv7":      ("YOLOv8 Open Images — 601 класс",         "Open Images V7",    601, False),
    "yoloe":     ("YOLOE-26 — свои названия объектов",      "LVIS",              None, True),
}
MODEL_CATALOG = [
    # family, size, file,              map,  params, flops, cpu_ms
    ("yolo26", "n", "yolo26n.pt",       40.9, 2.4,  5.5,   38.9),
    ("yolo26", "s", "yolo26s.pt",       48.6, 9.5,  20.9,  87.2),
    ("yolo26", "m", "yolo26m.pt",       53.1, 20.4, 68.4,  220.0),
    ("yolo26", "l", "yolo26l.pt",       55.0, 24.8, 86.8,  286.2),
    ("yolo26", "x", "yolo26x.pt",       57.5, 55.7, 194.4, 525.8),
    ("yolo11", "n", "yolo11n.pt",       39.5, 2.6,  6.5,   56.1),
    ("yolo11", "s", "yolo11s.pt",       47.0, 9.4,  21.6,  90.0),
    ("yolo11", "m", "yolo11m.pt",       51.5, 20.1, 68.1,  183.2),
    ("yolo11", "l", "yolo11l.pt",       53.4, 25.3, 87.2,  238.6),
    ("yolo11", "x", "yolo11x.pt",       54.7, 56.9, 195.3, 462.8),
    ("yolov8", "n", "yolov8n.pt",       37.3, 3.2,  8.7,   80.4),
    ("yolov8", "s", "yolov8s.pt",       44.9, 11.2, 28.6,  128.4),
    ("yolov8", "m", "yolov8m.pt",       50.2, 25.9, 78.9,  234.7),
    ("yolov8", "l", "yolov8l.pt",       52.9, 43.7, 165.1, 375.2),
    ("yolov8", "x", "yolov8x.pt",       53.9, 68.2, 257.8, 479.1),
    ("oiv7",   "n", "yolov8n-oiv7.pt",  18.4, 3.5,  10.4,  142.4),
    ("oiv7",   "s", "yolov8s-oiv7.pt",  27.7, 11.4, 29.7,  183.1),
    ("oiv7",   "m", "yolov8m-oiv7.pt",  33.6, 26.2, 80.6,  408.5),
    ("oiv7",   "l", "yolov8l-oiv7.pt",  34.9, 44.1, 167.4, 596.9),
    ("oiv7",   "x", "yolov8x-oiv7.pt",  36.3, 68.7, 260.6, 860.6),
    ("yoloe",  "s", "yoloe-26s-seg.pt", 30.8, 10.7, 21.9,  None),
    ("yoloe",  "m", "yoloe-26m-seg.pt", 35.4, 21.3, 70.6,  None),
    ("yoloe",  "l", "yoloe-26l-seg.pt", 37.8, 25.5, 89.0,  None),
    ("yoloe",  "x", "yoloe-26x-seg.pt", 40.6, 55.2, 197.7, None),
]


def _catalog_entry(family, size, file, mAP, params, flops, cpu_ms):
    group, dataset, n_classes, open_vocab = FAMILIES[family]
    short = {"yolo26": "YOLO26", "yolo11": "YOLO11", "yolov8": "YOLOv8",
             "oiv7": "YOLOv8 Open Images", "yoloe": "YOLOE-26"}[family]
    return {
        "id": file, "family": group, "size": size, "custom": False,
        "title": f"{short} {SIZE_NAMES[size]}",
        "hint": SIZE_HINTS[size],
        "dataset": dataset, "classes": n_classes, "open_vocab": open_vocab,
        "map": mAP, "params": params, "flops": flops, "cpu_ms": cpu_ms,
    }


MODELS = {e["id"]: e for e in (_catalog_entry(*row) for row in MODEL_CATALOG)}


def is_open_vocab(model_id: str) -> bool:
    """Модель ищет объекты по произвольным текстовым названиям (YOLOE / YOLO-World)."""
    if model_id in MODELS:
        return MODELS[model_id]["open_vocab"]
    return model_id.startswith("yoloe") and "-pf" not in model_id or "-world" in model_id


def available_models(models_dir: Path) -> list[dict]:
    """Каталог + собственные модели (*.pt), положенные в папку models/."""
    items = [dict(m, downloaded=(Path(models_dir) / m["id"]).exists()) for m in MODELS.values()]
    for p in sorted(Path(models_dir).glob("*.pt")):
        if p.name not in MODELS:
            open_vocab = p.name.startswith(("yoloe", "yolov8s-world", "yolov8m-world", "yolov8l-world", "yolov8x-world"))
            items.append({"id": p.name, "family": "Свои модели", "size": "", "custom": True,
                          "title": p.stem, "hint": "собственная модель из папки models/",
                          "dataset": None, "classes": None, "open_vocab": open_vocab and "-pf" not in p.name,
                          "map": None, "params": None, "flops": None, "cpu_ms": None, "downloaded": True})
    return items


LANGUAGES = ("ru", "en")


@dataclass
class Settings:
    # Камера
    camera_source: str = "0"      # номер камеры (0, 1, ...), URL потока или "browser" (трансляция из браузера)
    width: int = 1280
    height: int = 720
    mirror: bool = False
    # Распознавание
    detect_enabled: bool = True
    model: str = "yolo26n.pt"
    device: str = "auto"          # auto | cpu | cuda:0 | mps
    confidence: float = 0.40
    iou: float = 0.50
    imgsz: int = 640
    classes: list = field(default_factory=list)  # пусто = искать все классы
    # Свои названия для моделей с произвольными классами (YOLOE): «кружка», «очки = glasses», «red car»
    custom_classes: list = field(default_factory=lambda: ["человек", "очки", "кружка", "телефон", "клавиатура"])
    # Отображение
    show_boxes: bool = True
    show_labels: bool = True
    show_confidence: bool = True
    show_counts: bool = True
    show_fps: bool = True
    box_thickness: int = 3
    language: str = "ru"
    jpeg_quality: int = 80


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


class SettingsStore:
    """Потокобезопасное хранилище настроек с сохранением на диск."""

    def __init__(self, path: Path, models_dir: Path | None = None):
        self.path = Path(path)
        self.models_dir = Path(models_dir) if models_dir else self.path.parent / "models"
        self._lock = threading.Lock()
        self._settings = Settings()
        if self.path.exists():
            try:
                self.update(json.loads(self.path.read_text(encoding="utf-8")), save=False)
            except (OSError, ValueError):
                pass  # повреждённый файл — работаем с настройками по умолчанию

    def get(self) -> Settings:
        with self._lock:
            return replace(self._settings, classes=list(self._settings.classes))

    def to_dict(self) -> dict:
        return asdict(self.get())

    def update(self, data: dict, save: bool = True) -> Settings:
        with self._lock:
            current = asdict(self._settings)
            for f in fields(Settings):
                if f.name not in data:
                    continue
                try:
                    current[f.name] = self._coerce(f.name, data[f.name], current[f.name])
                except (TypeError, ValueError):
                    continue  # некорректное значение игнорируем
            self._settings = Settings(**current)
            if save:
                self.path.write_text(json.dumps(current, ensure_ascii=False, indent=2), encoding="utf-8")
            return replace(self._settings, classes=list(self._settings.classes))

    def _coerce(self, name, value, old):
        if isinstance(old, bool):
            if isinstance(value, str):
                return value.lower() in ("1", "true", "yes", "on")
            return bool(value)
        if isinstance(old, int):
            value = int(float(value))
        elif isinstance(old, float):
            value = float(value)
        elif isinstance(old, list):
            if not isinstance(value, list):
                raise ValueError
            if name == "custom_classes":  # порядок важен, дубликаты и пустые убираем
                seen, out = set(), []
                for v in value:
                    v = " ".join(str(v).split())[:80]
                    if v and v.lower() not in seen:
                        seen.add(v.lower())
                        out.append(v)
                return out[:100]
            return sorted({str(v) for v in value})
        else:
            value = str(value).strip()

        limits = {
            "confidence": (0.01, 1.0),
            "iou": (0.05, 1.0),
            "width": (160, 3840),
            "height": (120, 2160),
            "box_thickness": (1, 10),
            "jpeg_quality": (30, 95),
        }
        if name in limits:
            value = _clamp(value, *limits[name])
        if name == "imgsz":
            value = int(round(_clamp(value, 160, 1920) / 32) * 32)
        if name == "model" and value not in MODELS and not (
                value.endswith(".pt") and "/" not in value and "\\" not in value
                and (self.models_dir / value).exists()):
            raise ValueError
        if name == "device" and not re.fullmatch(r"auto|cpu|mps|cuda(:\d+)?", value):
            raise ValueError
        if name == "language" and value not in LANGUAGES:
            raise ValueError
        if name == "camera_source" and not value:
            raise ValueError
        return value
