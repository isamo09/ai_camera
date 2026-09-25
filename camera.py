"""Фоновый поток: получение кадров -> распознавание -> JPEG для веб-страницы.

Источники кадров:
  * камера компьютера или IP-камера — через OpenCV (cv2.VideoCapture);
  * браузер — страница сама присылает кадры (камера ноутбука/телефона или захват экрана).
"""
from __future__ import annotations

import os
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
        self.frame_source: str | None = None  # источник, с которого пришёл последний кадр
        self.open_index: int | None = None  # номер открытой камеры компьютера

        self._cap: cv2.VideoCapture | None = None
        self._opened_key = None
        self._file_interval = 0.0  # для видеофайлов — пауза между кадрами
        self._file_next = 0.0
        # Кадры, присылаемые браузером (камера телефона/ноутбука, захват экрана)
        self._push_cond = threading.Condition()
        self._pushed: np.ndarray | None = None
        self._push_id = 0
        self._push_seen = 0
        self._push_time = 0.0

    # ------------------------------------------------------------ public API
    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()
        with self._push_cond:
            self._push_cond.notify_all()
        self._thread.join(timeout=3)

    def detections(self) -> list[Detection]:
        with self._cond:
            return list(self._dets)

    def push_frame(self, data: bytes) -> bool:
        """Принимает JPEG/PNG-кадр от браузера."""
        frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            return False
        with self._push_cond:
            self._pushed = frame
            self._push_id += 1
            self._push_time = time.monotonic()
            self._push_cond.notify_all()
        return True

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

    def _release(self):
        if self._cap is not None:
            self._cap.release()
        self._cap, self._opened_key, self.open_index = None, None, None

    def _read_device(self, s) -> np.ndarray | None:
        """Кадр с камеры компьютера или IP-камеры (OpenCV)."""
        key = (s.camera_source, s.width, s.height)
        if self._cap is None or key != self._opened_key:
            self._release()
            self.status = "opening"
            cap = self._open(*key)
            if not cap.isOpened():
                cap.release()
                self.status = "error"
                self.error = f"Не удалось открыть камеру «{s.camera_source}»"
                self._placeholder(self.error + "\n" + camera_hint())
                self._stop.wait(2)
                return None
            self._cap, self._opened_key = cap, key
            self.open_index = int(s.camera_source) if s.camera_source.isdigit() else None
            # Видеофайл: воспроизводим по кругу с его собственной частотой кадров
            is_file = not s.camera_source.isdigit() and os.path.isfile(s.camera_source)
            fps = cap.get(cv2.CAP_PROP_FPS) if is_file else 0
            self._file_interval = 1.0 / fps if is_file and 0 < fps < 240 else 0.0
            self._file_next = time.perf_counter()
            if is_file:
                self.backend = "видеофайл"

        if self._file_interval:
            delay = self._file_next - time.perf_counter()
            if delay > 0:
                self._stop.wait(delay)
            self._file_next = max(self._file_next + self._file_interval, time.perf_counter())

        ok, frame = self._cap.read()
        if (not ok or frame is None) and self._file_interval:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # конец файла — с начала
            ok, frame = self._cap.read()
        if not ok or frame is None:
            self._release()
            self.status = "error"
            self.error = "Камера не отдаёт кадры — возможно, она занята или нет доступа. Переподключение…"
            self._placeholder("Камера не отдаёт кадры\n" + camera_hint())
            self._stop.wait(1)
            return None
        return frame

    def _read_browser(self) -> np.ndarray | None:
        """Кадр, присланный страницей (камера браузера, телефона или захват экрана)."""
        self._release()
        self.backend = "браузер"
        with self._push_cond:
            self._push_cond.wait_for(lambda: self._push_id != self._push_seen or self._stop.is_set(), timeout=1.0)
            if self._push_id == self._push_seen:
                frame = None
            else:
                frame, self._push_seen = self._pushed, self._push_id
        if frame is None and time.monotonic() - self._push_time > 2:
            self.status = "waiting_browser"
            self.error = None
            self.fps = 0.0
            self._placeholder("Ожидание трансляции из браузера…\n"
                              "Откройте панель «Источник видео» и нажмите «Начать трансляцию»")
        return frame

    def _loop(self):
        last_t = time.perf_counter()

        while not self._stop.is_set():
            s = self.store.get()

            # Модель: подгружаем/меняем, если выбрана другая модель или устройство
            if s.detect_enabled and self.detector.needs_reload(s.model, s.device):
                self.status = "loading_model"
                self._placeholder(f"Загрузка модели {s.model}…\nПри первом выборе веса скачиваются из интернета")
                try:
                    self.detector.ensure_model(s.model, s.device)
                    self.error = None
                except Exception as e:  # noqa: BLE001
                    self.error = f"Ошибка загрузки модели: {e}"
                    self._placeholder(f"Не удалось загрузить модель {s.model}\n{model_download_hint()}")
                    self._stop.wait(3)
                    continue

            frame = self._read_browser() if s.camera_source == "browser" else self._read_device(s)
            if frame is None:
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
            self.frame_source = s.camera_source
            self.frame_size = (frame.shape[1], frame.shape[0])
            annotated = annotate(frame, dets, s, self.fps)
            self._publish(annotated, frame, dets, s.jpeg_quality)

        self._release()
