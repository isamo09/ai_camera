"""Настройки приложения: хранятся в settings.json и меняются из веб-интерфейса."""
from __future__ import annotations

import json
import re
import threading
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path

# Каталог моделей. Скачиваются автоматически при первом выборе.
# Метрики — официальные данные Ultralytics для COCO val2017 при входе 640 px:
#   map  — mAP50-95 (%), params — млн параметров, flops — млрд операций на кадр,
#   cpu_ms — скорость на CPU (ONNX, Intel Xeon), по данным Ultralytics.
MODEL_CATALOG = [
    # family,   size, map,  params, flops, cpu_ms
    ("YOLO26", "n", 40.9, 2.4, 5.5, 38.9),
    ("YOLO26", "s", 48.6, 9.5, 20.9, 87.2),
    ("YOLO26", "m", 53.1, 20.4, 68.4, 220.0),
    ("YOLO26", "l", 55.0, 24.8, 86.8, 286.2),
    ("YOLO26", "x", 57.5, 55.7, 194.4, 525.8),
    ("YOLO11", "n", 39.5, 2.6, 6.5, 56.1),
    ("YOLO11", "s", 47.0, 9.4, 21.6, 90.0),
    ("YOLO11", "m", 51.5, 20.1, 68.1, 183.2),
    ("YOLO11", "l", 53.4, 25.3, 87.2, 238.6),
    ("YOLO11", "x", 54.7, 56.9, 195.3, 462.8),
    ("YOLOv8", "n", 37.3, 3.2, 8.7, 80.4),
    ("YOLOv8", "s", 44.9, 11.2, 28.6, 128.4),
    ("YOLOv8", "m", 50.2, 25.9, 78.9, 234.7),
    ("YOLOv8", "l", 52.9, 43.7, 165.1, 375.2),
    ("YOLOv8", "x", 53.9, 68.2, 257.8, 479.1),
]
SIZE_NAMES = {"n": "Nano", "s": "Small", "m": "Medium", "l": "Large", "x": "Extra Large"}
SIZE_HINTS = {
    "n": "самая быстрая, для любого процессора",
    "s": "баланс скорости и точности",
    "m": "точнее, лучше с видеокартой",
    "l": "высокая точность, нужна видеокарта",
    "x": "максимальная точность, мощная видеокарта",
}


def _catalog_entry(family, size, mAP, params, flops, cpu_ms):
    file = f"{family.lower()}{size}.pt"
    return {
        "id": file, "family": family, "size": size, "custom": False,
        "title": f"{family} {SIZE_NAMES[size]}",
        "hint": SIZE_HINTS[size],
        "map": mAP, "params": params, "flops": flops, "cpu_ms": cpu_ms,
    }


MODELS = {e["id"]: e for e in (_catalog_entry(*row) for row in MODEL_CATALOG)}


def available_models(models_dir: Path) -> list[dict]:
    """Каталог + собственные модели (*.pt), положенные в папку models/."""
    items = [dict(m, downloaded=(Path(models_dir) / m["id"]).exists()) for m in MODELS.values()]
    for p in sorted(Path(models_dir).glob("*.pt")):
        if p.name not in MODELS:
            items.append({"id": p.name, "family": "Свои модели", "size": "", "custom": True,
                          "title": p.stem, "hint": "собственная модель из папки models/",
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
