"""Фоновый поток: захват кадров с камеры (OpenCV) -> распознавание -> JPEG для веб-страницы."""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass

import cv2
import numpy as np

from detector import Detection, Detector, annotate, placeholder_frame
from platform_info import camera_backends, camera_hint, model_download_hint
from settings import SettingsStore


@dataclass
class Snapshot:
    annotated: np.ndarray
    raw: np.ndarray
    detections: list[Detection]


class CameraWorker:
    def __init__(self, store: SettingsStore, detector: Detector):
        self.store = store
        self.detector = detector
        self._cond = threading.Condition()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="camera", daemon=True)

        self._jpeg: bytes | None = None
        self._frame_id = 0
        self._raw: np.ndarray | None = None
        self._annotated: np.ndarray | None = None
        self._dets: list[Detection] = []

        self.fps = 0.0
        self.status = "starting"
        self.error: str | None = None
        self.frame_size = (0, 0)
        self.backend = ""

    # ------------------------------------------------------------ public API
    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=3)

    def detections(self) -> list[Detection]:
        with self._cond:
            return list(self._dets)

    def snapshot(self) -> Snapshot | None:
        with self._cond:
            if self._raw is None or self._annotated is None:
                return None
            return Snapshot(self._annotated.copy(), self._raw.copy(), list(self._dets))

    def mjpeg(self):
        """Генератор multipart/x-mixed-replace для тега <img>."""
        last_id = -1
        while not self._stop.is_set():
            with self._cond:
                self._cond.wait_for(lambda: self._frame_id != last_id, timeout=5)
                jpeg, last_id = self._jpeg, self._frame_id
            if jpeg is None:
                continue
            yield (b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                   + str(len(jpeg)).encode() + b"\r\n\r\n" + jpeg + b"\r\n")

    # ------------------------------------------------------------ internals
    def _open(self, source: str, width: int, height: int) -> cv2.VideoCapture:
        cap = None
        if source.isdigit():
            # Перебираем способы доступа к камере, подходящие для текущей ОС
            for backend, title in camera_backends():
                cap = cv2.VideoCapture(int(source), backend)
                if cap.isOpened():
                    self.backend = title
                    break
                cap.release()
        else:
            cap = cv2.VideoCapture(source)  # URL IP-камеры или видеофайл
            self.backend = "поток"
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    def _publish(self, frame: np.ndarray, raw: np.ndarray | None, dets: list[Detection], quality: int):
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not ok:
            return
        with self._cond:
            self._jpeg = buf.tobytes()
            self._annotated = frame if raw is not None else None
            self._raw = raw
            self._dets = dets
            self._frame_id += 1
            self._cond.notify_all()

    def _placeholder(self, text: str):
        s = self.store.get()
        self._publish(placeholder_frame(text, 1280, 720), None, [], s.jpeg_quality)

    def _loop(self):
        cap = None
        opened_key = None
        last_t = time.perf_counter()

        while not self._stop.is_set():
            s = self.store.get()

            # Модель: подгружаем/меняем, если выбрана другая
            if s.detect_enabled and self.detector.model_name != s.model:
                self.status = "loading_model"
                self._placeholder(f"Загрузка модели {s.model}…\nПри первом запуске веса скачиваются из интернета")
                try:
                    self.detector.ensure_model(s.model)
                    self.error = None
                except Exception as e:  # noqa: BLE001
                    self.error = f"Ошибка загрузки модели: {e}"
                    self._placeholder(f"Не удалось загрузить модель {s.model}\n{model_download_hint()}")
                    time.sleep(3)
                    continue

            # Камера: (пере)открываем при смене источника или разрешения
            key = (s.camera_source, s.width, s.height)
            if cap is None or key != opened_key:
                if cap is not None:
                    cap.release()
                self.status = "opening"
                cap = self._open(*key)
                opened_key = key
                if not cap.isOpened():
                    cap.release()
                    cap = None
                    self.status = "error"
                    self.error = f"Не удалось открыть камеру «{s.camera_source}»"
                    self._placeholder(self.error + "\n" + camera_hint())
                    self._stop.wait(2)
                    continue

            ok, frame = cap.read()
            if not ok or frame is None:
                cap.release()
                cap = None
                self.status = "error"
                self.error = "Камера не отдаёт кадры — возможно, она занята или нет доступа. Переподключение…"
                self._placeholder("Камера не отдаёт кадры\n" + camera_hint())
                self._stop.wait(1)
                continue

            if s.mirror:
                frame = cv2.flip(frame, 1)

            dets: list[Detection] = []
            if s.detect_enabled:
                try:
                    dets = self.detector.detect(frame, s)
                    self.error = None
                except Exception as e:  # noqa: BLE001
                    self.error = f"Ошибка распознавания: {e}"

            now = time.perf_counter()
            dt = now - last_t
            last_t = now
            if dt > 0:
                self.fps = 1 / dt if self.fps == 0 else self.fps * 0.9 + (1 / dt) * 0.1

            self.status = "running"
            self.frame_size = (frame.shape[1], frame.shape[0])
            annotated = annotate(frame, dets, s, self.fps)
            self._publish(annotated, frame, dets, s.jpeg_quality)

        if cap is not None:
            cap.release()
