"""AI-камера: распознавание объектов с веб-камеры в браузере.

Одинаково запускается на Windows, macOS и Linux:
    python main.py              (Windows)
    python3 main.py             (macOS / Linux)

При первом запуске сам создаёт .venv и ставит зависимости, выбирает свободный порт,
лучшее вычислительное устройство (NVIDIA / Apple Silicon / процессор) и открывает браузер.
"""
from __future__ import annotations

import os
import sys

# Консоли Windows/macOS/Linux могут иметь разную кодировку — не падаем на символах вроде «→»
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")
# На Apple Silicon операции, которых нет в Metal, выполнятся на процессоре, а не упадут
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

if __name__ == "__main__":
    import bootstrap

    bootstrap.ensure_environment()  # до импорта сторонних библиотек

import argparse
import threading
import time
import webbrowser
from pathlib import Path

from flask import Flask, Response, abort, jsonify, render_template, request, send_from_directory

import platform_info as pi
from camera import CameraWorker
from detector import FONT_PATH, Detector, summarize
from gallery import Gallery
from labels_ru import translate
from settings import SettingsStore, available_models

BASE_DIR = Path(__file__).resolve().parent

MODELS_DIR = BASE_DIR / "models"
store = SettingsStore(BASE_DIR / "settings.json", MODELS_DIR)
detector = Detector(MODELS_DIR, store.get().device)
gallery = Gallery(BASE_DIR / "gallery")
worker = CameraWorker(store, detector)

app = Flask(__name__)
app.json.ensure_ascii = False
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # кадры из браузера


# ---------------------------------------------------------------- страницы
@app.get("/")
def index():
    return render_template("index.html", page="camera")


@app.get("/gallery")
def gallery_page():
    return render_template("gallery.html", page="gallery")


@app.get("/video_feed")
def video_feed():
    return Response(worker.mjpeg(), mimetype="multipart/x-mixed-replace; boundary=frame",
                    headers={"Cache-Control": "no-cache, no-store"})


@app.get("/media/<path:filename>")
def media(filename):
    return send_from_directory(gallery.folder, filename)


# ---------------------------------------------------------------- API
@app.get("/api/status")
def api_status():
    s = store.get()
    dets = worker.detections()
    return jsonify(
        # пока не пришёл первый кадр с нового источника — «подключение», а не старый статус
        status=worker.status if worker.frame_source in (None, s.camera_source) or worker.status != "running" else "opening",
        error=worker.error,
        fps=round(worker.fps, 1),
        frame_size=worker.frame_size,
        model=detector.model_name,
        device=detector.device_title,
        device_id=detector.device,
        source=s.camera_source,
        backend=worker.backend,
        os=pi.OS_NAME,
        total=len(dets),
        counts=summarize(dets, s.language),
        objects=[d.to_dict(s.language) for d in dets],
        gallery_count=gallery.count(),
    )


@app.route("/api/settings", methods=["GET", "POST"])
def api_settings():
    if request.method == "POST":
        store.update(request.get_json(silent=True) or {})
    return jsonify(store.to_dict())


@app.get("/api/models")
def api_models():
    return jsonify(available_models(MODELS_DIR))


@app.get("/api/devices")
def api_devices():
    """Вычислительные устройства компьютера: авто, процессор, видеокарты NVIDIA, Apple Silicon."""
    return jsonify(pi.list_devices())


_cameras_cache: dict = {"time": 0.0, "items": []}


@app.get("/api/cameras")
def api_cameras():
    """Камеры, подключённые к компьютеру (кэш 10 с, ?refresh=1 — обновить)."""
    now = time.monotonic()
    if request.args.get("refresh") or now - _cameras_cache["time"] > 10:
        skip = {worker.open_index} if worker.open_index is not None else set()
        _cameras_cache.update(time=now, items=pi.list_cameras(skip_probe=skip))
    return jsonify(_cameras_cache["items"])


@app.post("/api/push_frame")
def api_push_frame():
    """Кадр от браузера (камера устройства или захват экрана) в формате JPEG."""
    if store.get().camera_source != "browser":
        return jsonify(error="Источник видео — не браузер"), 409
    if not worker.push_frame(request.get_data()):
        return jsonify(error="Не удалось прочитать изображение"), 400
    return ("", 204)


@app.get("/api/classes")
def api_classes():
    lang = store.get().language
    return jsonify([{"id": i, "name": n, "label": translate(n, lang)}
                    for i, n in sorted(detector.names.items())])


@app.post("/api/snapshot")
def api_snapshot():
    data = request.get_json(silent=True) or {}
    snap = worker.snapshot()
    if snap is None:
        return jsonify(error="Нет кадра с камеры — снимок сделать нельзя"), 503
    meta = gallery.save(snap.annotated, snap.raw, snap.detections,
                        note=data.get("note", ""), lang=store.get().language)
    return jsonify(meta)


@app.get("/api/gallery")
def api_gallery():
    return jsonify(gallery.list())


@app.route("/api/gallery/<sid>", methods=["PATCH", "DELETE"])
def api_gallery_item(sid):
    if request.method == "DELETE":
        if not gallery.delete(sid):
            abort(404)
        return jsonify(ok=True)
    meta = gallery.update_note(sid, (request.get_json(silent=True) or {}).get("note", ""))
    if meta is None:
        abort(404)
    return jsonify(meta)


def main():
    parser = argparse.ArgumentParser(description="AI-камера с распознаванием объектов")
    parser.add_argument("--host", default="0.0.0.0",
                        help="адрес для прослушивания (по умолчанию 0.0.0.0 — доступ из локальной сети; "
                             "127.0.0.1 — только с этого компьютера)")
    parser.add_argument("--port", type=int, default=pi.default_port(),
                        help=f"порт (по умолчанию {pi.default_port()}; если занят — берётся следующий свободный)")
    parser.add_argument("--camera", help="номер камеры или URL потока (только на этот запуск)")
    parser.add_argument("--device", help="auto | cpu | cuda | cuda:0 | mps — только на этот запуск "
                                           "(обычно выбирается на странице в настройках)")
    parser.add_argument("--https", action="store_true",
                        help="HTTPS с самоподписанным сертификатом — нужен для камеры телефона по локальной сети")
    parser.add_argument("--no-browser", action="store_true", help="не открывать браузер автоматически")
    args = parser.parse_args()

    if args.camera is not None:
        store.update({"camera_source": args.camera}, save=False)  # только на этот запуск
    if args.device:
        store.update({"device": args.device}, save=False)

    port = pi.find_free_port(args.host, args.port)
    if port != args.port:
        print(f"[server] Порт {args.port} занят, используется {port}")

    print(f"\n  Система:     {pi.os_description()}, Python {sys.version.split()[0]}")
    print(f"  Вычисления:  {pi.device_title(detector.resolve_device(store.get().device))}"
          f"  (настройка: {store.get().device})")
    print(f"  Шрифт:       {FONT_PATH or 'встроенный (без кириллицы — установите Arial или DejaVu Sans)'}")

    try:
        detector.ensure_model(store.get().model, store.get().device)  # при первом запуске скачает веса
    except Exception as e:  # noqa: BLE001
        print(f"[detector] Не удалось загрузить модель: {e}\n{pi.model_download_hint()}")
        print("[detector] Сервер всё равно запустится, модель будет загружена при следующей попытке.")
    worker.start()
    threading.Thread(target=pi.list_devices, daemon=True).start()  # заранее собираем список устройств

    ssl_context = None
    if args.https:
        try:
            import cryptography  # noqa: F401 — нужен Werkzeug для самоподписанного сертификата
            ssl_context = "adhoc"
        except ImportError:
            print("[server] Для --https нужен пакет cryptography: pip install cryptography. Запуск без HTTPS.")
    scheme = "https" if ssl_context else "http"

    host = "127.0.0.1" if args.host in ("0.0.0.0", "127.0.0.1") else args.host
    local_url = f"{scheme}://{host}:{port}"
    print(f"\n  Откройте в браузере:  {local_url}")
    if args.host == "0.0.0.0" and (ips := pi.lan_ips()):
        print(f"  С телефона / другого устройства в сети:  {scheme}://{ips[0]}:{port}")
        for other in ips[1:3]:
            print(f"                          или:              {scheme}://{other}:{port}")
        if not ssl_context:
            print("  (камера телефона через браузер работает только по HTTPS — запустите с --https)")
    if ssl_context:
        print("  Сертификат самоподписанный: в браузере нажмите «Дополнительно» → «Перейти на сайт».")
    print("  Остановить: Ctrl+C\n")

    if not args.no_browser and pi.can_open_browser():
        threading.Timer(1.5, lambda: webbrowser.open(local_url)).start()

    try:
        app.run(host=args.host, port=port, threaded=True, debug=False, use_reloader=False, ssl_context=ssl_context)
    finally:
        worker.stop()


if __name__ == "__main__":
    main()
