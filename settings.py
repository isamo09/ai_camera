"""Настройки приложения: хранятся в settings.json и меняются из веб-интерфейса."""
from __future__ import annotations

import json
import threading
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path

# Модели, доступные для выбора (скачиваются автоматически при первом использовании)
MODELS = {
    "yolo26n.pt": "YOLO26 Nano — новейшая, быстрая",
    "yolo26s.pt": "YOLO26 Small — новейшая, точнее",
    "yolo26m.pt": "YOLO26 Medium — новейшая, нужна мощная машина",
    "yolo11n.pt": "YOLO11 Nano — самая быстрая",
    "yolo11s.pt": "YOLO11 Small — баланс",
    "yolo11m.pt": "YOLO11 Medium — точнее, медленнее",
    "yolo11l.pt": "YOLO11 Large — нужна видеокарта",
    "yolov8n.pt": "YOLOv8 Nano — классическая",
}

LANGUAGES = ("ru", "en")


@dataclass
class Settings:
    # Камера
    camera_source: str = "0"      # номер камеры (0, 1, ...) или URL потока (rtsp://, http://)
    width: int = 1280
    height: int = 720
    mirror: bool = False
    # Распознавание
    detect_enabled: bool = True
    model: str = "yolo26n.pt"
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

    def __init__(self, path: Path):
        self.path = Path(path)
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

    @staticmethod
    def _coerce(name, value, old):
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
        if name == "model" and value not in MODELS:
            raise ValueError
        if name == "language" and value not in LANGUAGES:
            raise ValueError
        if name == "camera_source" and not value:
            raise ValueError
        return value
