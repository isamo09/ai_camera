"""Подготовка окружения при запуске: только стандартная библиотека Python.

Если нужных библиотек нет, создаёт виртуальное окружение .venv рядом с проектом,
устанавливает зависимости из requirements.txt и перезапускает main.py уже из него.
Работает одинаково на Windows, macOS и Linux.
"""
from __future__ import annotations

import importlib
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
VENV_DIR = BASE_DIR / ".venv"
REQUIREMENTS = BASE_DIR / "requirements.txt"
REQUIRED_MODULES = ("flask", "cv2", "ultralytics", "PIL", "numpy", "cv2_enumerate_cameras", "cryptography")
MIN_PYTHON = (3, 9)
FLAG = "AI_CAMERA_BOOTSTRAPPED"


def _venv_python() -> Path:
    if sys.platform.startswith("win"):
        return VENV_DIR / "Scripts" / "python.exe"
    return VENV_DIR / "bin" / "python"


def _missing_modules() -> list[str]:
    importlib.invalidate_caches()
    return [m for m in REQUIRED_MODULES if importlib.util.find_spec(m) is None]


def _in_virtualenv() -> bool:
    return sys.prefix != sys.base_prefix or bool(os.environ.get("CONDA_PREFIX"))


def _has_nvidia_gpu() -> bool:
    if sys.platform == "darwin":
        return False
    try:
        out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=10)
        return out.returncode == 0 and "GPU" in out.stdout
    except (OSError, subprocess.SubprocessError):
        return False


def _install_cuda_torch(python: Path | str) -> None:
    """По умолчанию pip ставит PyTorch без поддержки видеокарты (на Windows) — берём сборку с CUDA."""
    print("[setup] Найдена видеокарта NVIDIA — устанавливаю PyTorch с поддержкой CUDA (~2–3 ГБ)…", flush=True)
    for cuda in ("cu130", "cu128", "cu126", "cu124"):
        cmd = [str(python), "-m", "pip", "install", "--disable-pip-version-check", "torch", "torchvision",
               "--index-url", f"https://download.pytorch.org/whl/{cuda}"]
        if subprocess.call(cmd) == 0:
            return
    print("[setup] Сборку с CUDA установить не удалось — будет использоваться процессор.", flush=True)


def _pip_install(python: Path | str) -> None:
    print(f"[setup] Установка зависимостей ({REQUIREMENTS.name}) — это нужно только один раз…", flush=True)
    if _has_nvidia_gpu() and os.environ.get("AI_CAMERA_CPU_ONLY") != "1":
        _install_cuda_torch(python)
    cmd = [str(python), "-m", "pip", "install", "--disable-pip-version-check", "-r", str(REQUIREMENTS)]
    if subprocess.call(cmd) != 0:
        hint = ""
        if sys.platform.startswith("win"):
            hint = ("\nЕсли выше ошибка WinError 206 — слишком длинный путь к папке проекта: переместите её,\n"
                    "например, в C:\\ai_camera, удалите папку .venv и запустите снова.")
        sys.exit(f"[setup] Не удалось установить зависимости. Проверьте интернет и повторите запуск.{hint}")


def _create_venv() -> None:
    import venv

    print(f"[setup] Создаю виртуальное окружение: {VENV_DIR}", flush=True)
    try:
        venv.create(VENV_DIR, with_pip=True)
    except Exception as e:  # noqa: BLE001
        hint = ""
        if sys.platform.startswith("linux"):
            hint = "\nУстановите модуль venv: sudo apt install python3-venv (Debian/Ubuntu)."
        sys.exit(f"[setup] Не удалось создать виртуальное окружение: {e}{hint}")


def _check_opencv() -> None:
    """На «голых» Linux-системах OpenCV может требовать системную libGL."""
    try:
        import cv2  # noqa: F401
    except ImportError as e:
        msg = str(e)
        if "libGL" in msg or "libgthread" in msg:
            sys.exit("[setup] OpenCV требует системные библиотеки.\n"
                     "Debian/Ubuntu: sudo apt install libgl1 libglib2.0-0\n"
                     "Fedora: sudo dnf install mesa-libGL")
        raise


def ensure_environment() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    if sys.version_info < MIN_PYTHON:
        sys.exit(f"Нужен Python {'.'.join(map(str, MIN_PYTHON))} или новее, а запущен {sys.version.split()[0]}.")

    if not _missing_modules():
        _check_opencv()
        return

    # Уже внутри виртуального окружения (нашего или пользовательского) — ставим прямо сюда
    if _in_virtualenv() or os.environ.get(FLAG) == "1":
        _pip_install(sys.executable)
        if _missing_modules():
            sys.exit(f"[setup] После установки всё ещё нет модулей: {', '.join(_missing_modules())}")
        _check_opencv()
        return

    # Системный Python — готовим .venv и перезапускаемся из него
    venv_python = _venv_python()
    if not venv_python.exists():
        _create_venv()
    env = dict(os.environ, **{FLAG: "1"})
    print(f"[setup] Перезапуск через {venv_python}", flush=True)
    try:
        code = subprocess.call([str(venv_python), str(BASE_DIR / "main.py"), *sys.argv[1:]], env=env)
    except KeyboardInterrupt:  # Ctrl+C получает и дочерний процесс — он завершится сам
        code = 0
    sys.exit(code)
