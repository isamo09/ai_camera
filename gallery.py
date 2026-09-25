"""Галерея снимков: изображения + JSON с описанием в папке gallery/."""
from __future__ import annotations

import json
import re
import threading
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from detector import Detection, describe, summarize

ID_RE = re.compile(r"^\d{8}_\d{6}_\d{3}$")


class Gallery:
    def __init__(self, folder: Path):
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    @staticmethod
    def _write_jpeg(path: Path, img: np.ndarray, quality: int = 92):
        # imencode + write_bytes вместо cv2.imwrite: работает и с кириллицей в пути
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not ok:
            raise RuntimeError("Не удалось закодировать изображение")
        path.write_bytes(buf.tobytes())

    def save(self, annotated: np.ndarray, raw: np.ndarray, dets: list[Detection],
             note: str = "", lang: str = "ru") -> dict:
        with self._lock:
            now = datetime.now()
            sid = now.strftime("%Y%m%d_%H%M%S_%f")[:-3]
            self._write_jpeg(self.folder / f"{sid}.jpg", annotated)
            self._write_jpeg(self.folder / f"{sid}_raw.jpg", raw)
            h, w = raw.shape[:2]
            meta = {
                "id": sid,
                "created": now.isoformat(timespec="seconds"),
                "image": f"{sid}.jpg",
                "raw_image": f"{sid}_raw.jpg",
                "width": w,
                "height": h,
                "total": len(dets),
                "counts": summarize(dets, lang),
                "objects": [d.to_dict(lang) for d in dets],
                "description": describe(dets, lang),
                "note": str(note)[:500],
            }
            self._meta_path(sid).write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
            return meta

    def _meta_path(self, sid: str) -> Path:
        return self.folder / f"{sid}.json"

    def list(self) -> list[dict]:
        items = []
        for p in self.folder.glob("*.json"):
            try:
                items.append(json.loads(p.read_text(encoding="utf-8")))
            except (OSError, ValueError):
                continue
        items.sort(key=lambda m: m.get("id", ""), reverse=True)
        return items

    def count(self) -> int:
        return sum(1 for _ in self.folder.glob("*.json"))

    def update_note(self, sid: str, note: str) -> dict | None:
        if not ID_RE.match(sid):
            return None
        with self._lock:
            path = self._meta_path(sid)
            if not path.exists():
                return None
            meta = json.loads(path.read_text(encoding="utf-8"))
            meta["note"] = str(note)[:500]
            path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
            return meta

    def delete(self, sid: str) -> bool:
        if not ID_RE.match(sid):
            return False
        with self._lock:
            found = False
            for name in (f"{sid}.json", f"{sid}.jpg", f"{sid}_raw.jpg"):
                p = self.folder / name
                if p.exists():
                    p.unlink()
                    found = True
            return found
