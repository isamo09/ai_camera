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
import webbrowser
from pathlib import Path

from flask import Flask, Response, abort, jsonify, render_template, request, send_from_directory

import platform_info as pi
from camera import CameraWorker
from detector import FONT_PATH, Detector, summarize
from gallery import Gallery
from labels_ru import translate
from settings import MODELS, SettingsStore

BASE_DIR = Path(__file__).resolve().parent

store = SettingsStore(BASE_DIR / "settings.json")
detector = Detector(BASE_DIR / "models")
gallery = Gallery(BASE_DIR / "gallery")
worker = CameraWorker(store, detector)

app = Flask(__name__)
app.json.ensure_ascii = False


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
        status=worker.status,
        error=worker.error,
        fps=round(worker.fps, 1),
        frame_size=worker.frame_size,
        model=detector.model_name,
        device=detector.device_title,
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
    return jsonify([{"id": k, "title": v} for k, v in MODELS.items()])


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
    parser.add_argument("--host", default="127.0.0.1", help="адрес (0.0.0.0 — доступ из локальной сети)")
    parser.add_argument("--port", type=int, default=pi.default_port(),
                        help=f"порт (по умолчанию {pi.default_port()}; если занят — берётся следующий свободный)")
    parser.add_argument("--camera", help="номер камеры или URL потока (только на этот запуск)")
    parser.add_argument("--device", default="auto",
                        help="auto | cpu | cuda | cuda:0 | mps  (по умолчанию выбирается автоматически)")
    parser.add_argument("--no-browser", action="store_true", help="не открывать браузер автоматически")
    args = parser.parse_args()

    if args.camera is not None:
        store.update({"camera_source": args.camera}, save=False)  # только на этот запуск
    detector.set_device(args.device)

    port = pi.find_free_port(args.host, args.port)
    if port != args.port:
        print(f"[server] Порт {args.port} занят, используется {port}")

    print(f"\n  Система:     {pi.os_description()}, Python {sys.version.split()[0]}")
    print(f"  Вычисления:  {detector.device_title}")
    print(f"  Шрифт:       {FONT_PATH or 'встроенный (без кириллицы — установите Arial или DejaVu Sans)'}")

    try:
        detector.ensure_model(store.get().model)  # при первом запуске скачает веса (~5 МБ)
    except Exception as e:  # noqa: BLE001
        print(f"[detector] Не удалось загрузить модель: {e}\n{pi.model_download_hint()}")
        print("[detector] Сервер всё равно запустится, модель будет загружена при следующей попытке.")
    worker.start()

    local_url = f"http://127.0.0.1:{port}" if args.host in ("0.0.0.0", "127.0.0.1") else f"http://{args.host}:{port}"
    print(f"\n  Откройте в браузере:  {local_url}")
    if args.host == "0.0.0.0" and (ip := pi.lan_ip()):
        print(f"  С телефона / другого устройства в сети:  http://{ip}:{port}")
    print("  Остановить: Ctrl+C\n")

    if not args.no_browser and pi.can_open_browser():
        threading.Timer(1.5, lambda: webbrowser.open(local_url)).start()

    try:
        app.run(host=args.host, port=port, threaded=True, debug=False, use_reloader=False)
    finally:
        worker.stop()


if __name__ == "__main__":
    main()
