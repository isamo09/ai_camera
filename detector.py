"""Распознавание объектов (YOLO / Ultralytics) и отрисовка разметки на кадре."""
from __future__ import annotations

import colorsys
import threading
from collections import OrderedDict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from labels_ru import CUSTOM_LABELS, parse_custom, translate
from platform_info import device_title, find_font, pick_device
from settings import Settings, is_open_vocab

# Шрифт с кириллицей под текущую ОС (cv2.putText русские буквы не умеет, поэтому рисуем через Pillow)
FONT_PATH = find_font()

POSITION_WORDS = {
    "ru": {
        "v": {"top": "вверху", "middle": "посередине", "bottom": "внизу"},
        "h": {"left": "слева", "center": "по центру", "right": "справа"},
        "center": "в центре",
    },
    "en": {
        "v": {"top": "top", "middle": "middle", "bottom": "bottom"},
        "h": {"left": "left", "center": "center", "right": "right"},
        "center": "center",
    },
}


@lru_cache(maxsize=32)
def get_font(size: int) -> ImageFont.ImageFont:
    if FONT_PATH:
        return ImageFont.truetype(FONT_PATH, size)
    return ImageFont.load_default(size)


def class_color(cls_id: int) -> tuple[int, int, int]:
    """Стабильный яркий цвет для каждого класса (RGB)."""
    hue = (cls_id * 0.618033988749895 + 0.08) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.72, 1.0)
    return int(r * 255), int(g * 255), int(b * 255)


def to_hex(rgb) -> str:
    return "#{:02x}{:02x}{:02x}".format(*rgb)


def _text_color(rgb) -> tuple[int, int, int]:
    r, g, b = rgb
    return (0, 0, 0) if (0.299 * r + 0.587 * g + 0.114 * b) > 150 else (255, 255, 255)


def position_text(key: str, lang: str) -> str:
    v, h = key.split("-")
    words = POSITION_WORDS.get(lang, POSITION_WORDS["ru"])
    if v == "middle" and h == "center":
        return words["center"]
    if v == "middle":
        return words["h"][h]
    return f'{words["v"][v]} {words["h"][h]}'


@dataclass
class Detection:
    cls_id: int
    name: str                      # исходное (английское) имя класса
    conf: float
    box: tuple[int, int, int, int]  # x1, y1, x2, y2
    position: str                  # например "top-left"

    @property
    def color(self):
        return class_color(self.cls_id)

    def label(self, lang: str) -> str:
        return translate(self.name, lang)

    def to_dict(self, lang: str) -> dict:
        return {
            "name": self.name,
            "label": self.label(lang),
            "conf": round(self.conf, 3),
            "box": list(self.box),
            "position": self.position,
            "position_text": position_text(self.position, lang),
            "color": to_hex(self.color),
        }


class Detector:
    def __init__(self, models_dir: Path, device: str = "auto"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self.model = None
        self.model_name: str | None = None
        self.names: dict[int, str] = {}
        self.open_vocab = False
        self.vocab: tuple[str, ...] | None = None
        self._resolved: dict[str, str] = {}
        self.requested_device = "auto"
        self.device = self.resolve_device(device)
        self.device_title = device_title(self.device)

    def resolve_device(self, requested: str) -> str:
        """auto → видеокарта NVIDIA (cuda) → графика Apple Silicon (mps) → процессор."""
        if requested not in self._resolved:
            self._resolved[requested] = pick_device(requested)
        return self._resolved[requested]

    def needs_reload(self, name: str, device: str) -> bool:
        return self.model is None or self.model_name != name or self.device != self.resolve_device(device)

    def ensure_model(self, name: str, device: str | None = None) -> None:
        """Загружает модель на нужное устройство (при первом запуске — скачивает веса)."""
        device = self.requested_device if device is None else device
        if not self.needs_reload(name, device):
            return
        with self._lock:
            if not self.needs_reload(name, device):
                return
            from ultralytics import YOLO, YOLOE  # тяжёлый импорт — только когда нужно

            target = self.resolve_device(device)
            print(f"[detector] Загрузка модели {name} на {device_title(target)} ...")
            cls = YOLOE if name.startswith("yoloe") else YOLO
            model = cls(str(self.models_dir / name))
            model.to(target)
            self.model, self.model_name = model, name
            self.requested_device, self.device = device, target
            self.device_title = device_title(target)
            self.open_vocab = is_open_vocab(name)
            self.vocab = None
            CUSTOM_LABELS.clear()
            self.names = {int(k): v for k, v in model.names.items()}
            print(f"[detector] Модель {name} готова"
                  + (" (свои названия объектов)" if self.open_vocab else f", классов: {len(self.names)}"))

    # ------------------------------------------------------------ свои названия (YOLOE)
    @staticmethod
    def vocab_for(entries: list[str]) -> tuple[list[str], dict[str, str]]:
        """Запросы для модели и подписи к ним из введённых пользователем названий."""
        prompts, labels = [], {}
        for e in entries:
            p = parse_custom(e)
            if p["prompt"] and p["prompt"] not in labels:
                prompts.append(p["prompt"])
                labels[p["prompt"]] = p["label"]
        return prompts, labels

    def needs_vocab(self, s: Settings) -> bool:
        return self.open_vocab and self.model is not None and \
            tuple(self.vocab_for(s.custom_classes)[0]) != self.vocab

    def ensure_vocab(self, entries: list[str]) -> None:
        """Передаёт модели список названий. Первый раз скачивает текстовый кодировщик (~250 МБ)."""
        prompts, labels = self.vocab_for(entries)
        if not self.open_vocab or tuple(prompts) == self.vocab:
            return
        with self._lock:
            if prompts:
                print(f"[detector] Свои названия: {', '.join(prompts)}")
                self.model.set_classes(prompts)
            CUSTOM_LABELS.clear()
            CUSTOM_LABELS.update(labels)
            self.names = dict(enumerate(prompts))
            self.vocab = tuple(prompts)

    def detect(self, frame: np.ndarray, s: Settings) -> list[Detection]:
        self.ensure_model(s.model, s.device)
        if self.open_vocab:
            self.ensure_vocab(s.custom_classes)
            if not self.vocab:
                return []  # список своих названий пуст — искать нечего
        class_ids = None
        if s.classes:
            wanted = set(s.classes)
            class_ids = [i for i, n in self.names.items() if n in wanted] or None

        result = self.model.predict(
            frame, conf=s.confidence, iou=s.iou, imgsz=s.imgsz,
            classes=class_ids, device=self.device, verbose=False,
        )[0]

        h, w = frame.shape[:2]
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return []
        xyxy = boxes.xyxy.cpu().numpy().astype(int)
        cls = boxes.cls.cpu().numpy().astype(int)
        conf = boxes.conf.cpu().numpy()

        detections = []
        for (x1, y1, x2, y2), c, p in zip(xyxy, cls, conf):
            cx, cy = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h
            v = "top" if cy < 1 / 3 else "bottom" if cy > 2 / 3 else "middle"
            hz = "left" if cx < 1 / 3 else "right" if cx > 2 / 3 else "center"
            detections.append(Detection(int(c), self.names.get(int(c), str(c)), float(p),
                                        (int(x1), int(y1), int(x2), int(y2)), f"{v}-{hz}"))
        detections.sort(key=lambda d: d.conf, reverse=True)
        return detections


# ---------------------------------------------------------------- сводки и описания

def summarize(dets: list[Detection], lang: str) -> list[dict]:
    """Количество объектов каждого класса, по убыванию."""
    groups: "OrderedDict[str, dict]" = OrderedDict()
    for d in dets:
        g = groups.setdefault(d.name, {"name": d.name, "label": d.label(lang),
                                       "count": 0, "color": to_hex(d.color)})
        g["count"] += 1
    return sorted(groups.values(), key=lambda g: (-g["count"], g["label"]))


def describe(dets: list[Detection], lang: str) -> str:
    """Текстовое описание кадра: что, сколько и где."""
    if not dets:
        return "Объекты не обнаружены." if lang == "ru" else "No objects detected."
    groups = summarize(dets, lang)
    if lang == "ru":
        lines = [f"Обнаружено объектов: {len(dets)} (видов: {len(groups)})."]
    else:
        lines = [f"Objects detected: {len(dets)} ({len(groups)} kinds)."]
    for g in groups:
        where = ", ".join(
            f"{position_text(d.position, lang)} ({d.conf:.0%})" for d in dets if d.name == g["name"]
        )
        lines.append(f"• {g['label']} × {g['count']} — {where}")
    return "\n".join(lines)


# ---------------------------------------------------------------- отрисовка

def annotate(frame: np.ndarray, dets: list[Detection], s: Settings, fps: float | None = None) -> np.ndarray:
    """Рисует рамки, подписи, счётчик объектов и FPS. Возвращает новый BGR-кадр."""
    h, w = frame.shape[:2]
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img, "RGBA")
    base = max(14, int(min(w, h) / 30))
    label_font = get_font(max(12, int(base * 0.8)))
    thickness = max(1, int(s.box_thickness * max(w, h) / 1280)) if s.box_thickness else 1

    for d in dets:
        x1, y1, x2, y2 = d.box
        color = d.color
        if s.show_boxes:
            draw.rectangle([x1, y1, x2, y2], outline=color, width=thickness)
        if s.show_labels:
            text = d.label(s.language)
            if s.show_confidence:
                text += f" {d.conf:.0%}"
            l, t, r, b = draw.textbbox((0, 0), text, font=label_font)
            tw, th = r - l, b - t
            pad = max(3, th // 4)
            ly = y1 - th - 2 * pad if y1 - th - 2 * pad >= 0 else y1
            lx = min(max(0, x1), max(0, w - tw - 2 * pad))
            draw.rectangle([lx, ly, lx + tw + 2 * pad, ly + th + 2 * pad], fill=color + (230,))
            draw.text((lx + pad - l, ly + pad - t), text, font=label_font, fill=_text_color(color))

    if s.show_counts:
        _draw_counts(draw, dets, s.language, base)
    if s.show_fps and fps is not None:
        font = get_font(max(12, int(base * 0.75)))
        text = f"{fps:.1f} FPS"
        l, t, r, b = draw.textbbox((0, 0), text, font=font)
        pad = 8
        x = w - (r - l) - 3 * pad
        draw.rounded_rectangle([x, pad, w - pad, pad + (b - t) + 2 * pad], radius=8, fill=(0, 0, 0, 150))
        draw.text((x + pad - l, 2 * pad - t), text, font=font, fill=(255, 255, 255))

    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


def _draw_counts(draw: ImageDraw.ImageDraw, dets, lang, base):
    groups = summarize(dets, lang)
    title_font = get_font(max(13, int(base * 0.85)))
    font = get_font(max(12, int(base * 0.72)))
    title = (f"В кадре: {len(dets)}" if lang == "ru" else f"In frame: {len(dets)}")
    lines = [(f"{g['label']}: {g['count']}", g["color"]) for g in groups[:12]]
    if len(groups) > 12:
        lines.append(("…", "#ffffff"))

    pad, gap = 10, 6
    sizes = [draw.textbbox((0, 0), t, font=font) for t, _ in lines]
    tb = draw.textbbox((0, 0), title, font=title_font)
    line_h = max([b - t for _, t, _, b in sizes] + [10])
    sw = line_h  # квадратик-цвет
    width = max([tb[2] - tb[0]] + [(r - l) + sw + gap for l, _, r, _ in sizes]) + 2 * pad
    height = (tb[3] - tb[1]) + 2 * pad + len(lines) * (line_h + gap)

    x0, y0 = 8, 8
    draw.rounded_rectangle([x0, y0, x0 + width, y0 + height], radius=10, fill=(0, 0, 0, 160))
    draw.text((x0 + pad - tb[0], y0 + pad - tb[1]), title, font=title_font, fill=(255, 255, 255))
    y = y0 + pad + (tb[3] - tb[1]) + gap + 2
    for (text, color), (l, t, _, _) in zip(lines, sizes):
        draw.rounded_rectangle([x0 + pad, y + 2, x0 + pad + sw - 4, y + sw - 2], radius=3, fill=color)
        draw.text((x0 + pad + sw + gap - l - 4, y - t), text, font=font, fill=(235, 235, 235))
        y += line_h + gap


def placeholder_frame(text: str, width: int = 1280, height: int = 720) -> np.ndarray:
    """Заглушка, когда камера недоступна или загружается модель."""
    img = Image.new("RGB", (width, height), (18, 20, 26))
    draw = ImageDraw.Draw(img)
    lines = text.split("\n")
    size = max(18, height // 22)
    while True:  # уменьшаем шрифт, пока самая длинная строка не поместится
        font = get_font(size)
        boxes = [draw.textbbox((0, 0), ln, font=font) for ln in lines]
        if size <= 12 or max(b[2] - b[0] for b in boxes) <= width * 0.92:
            break
        size -= 2
    total = sum(b[3] - b[1] + 12 for b in boxes)
    y = (height - total) // 2
    for ln, (l, t, r, b) in zip(lines, boxes):
        draw.text(((width - (r - l)) // 2 - l, y - t), ln, font=font, fill=(200, 205, 215))
        y += b - t + 12
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
