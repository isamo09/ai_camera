"""Подготовка окружения при запуске: только стандартная библиотека Python.

Если нужных библиотек нет, создаёт виртуальное окружение .venv рядом с проектом,
устанавливает зависимости из requirements.txt и перезапускает main.py уже из него.
Работает одинаково на Windows, macOS и Linux.
"""
from __future__ import annotations

import importlib
import importlib.util
import os
import re
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


# ---------------------------------------------------------------- видеокарта NVIDIA (Windows и Linux)

# Сборки PyTorch с CUDA и минимальная версия CUDA, которую должен поддерживать драйвер.
# Новый драйвер умеет запускать и более старые сборки, поэтому берём самую новую подходящую.
CUDA_BUILDS = [("cu130", (13, 0)), ("cu128", (12, 8)), ("cu126", (12, 6)),
               ("cu124", (12, 4)), ("cu121", (12, 1)), ("cu118", (11, 8))]
TORCH_INDEX = "https://download.pytorch.org/whl/"


def nvidia_smi(*args: str) -> str | None:
    """Вывод nvidia-smi или None, если драйвера NVIDIA нет. Учитывает WSL2 и стандартные пути Windows."""
    if sys.platform == "darwin":
        return None
    candidates = ["nvidia-smi", "/usr/lib/wsl/lib/nvidia-smi", "/usr/bin/nvidia-smi",
                  r"C:\Windows\System32\nvidia-smi.exe"]
    for exe in candidates:
        try:
            out = subprocess.run([exe, *args], capture_output=True, text=True, timeout=15)
            if out.returncode == 0 and out.stdout.strip():
                return out.stdout
        except (OSError, subprocess.SubprocessError):
            continue
    return None


def driver_cuda_version() -> tuple[int, int] | None:
    """Максимальная версия CUDA, которую поддерживает установленный драйвер (из шапки nvidia-smi)."""
    out = nvidia_smi()
    m = re.search(r"CUDA(?: UMD)? Version\s*:\s*(\d+)\.(\d+)", out or "")
    return (int(m.group(1)), int(m.group(2))) if m else None


def linux_nvidia_hardware() -> bool:
    """Linux: есть ли видеокарта NVIDIA на шине PCI (даже если драйвер не установлен)."""
    if not sys.platform.startswith("linux"):
        return False
    for dev in Path("/sys/bus/pci/devices").glob("*"):
        try:
            vendor = (dev / "vendor").read_text().strip()
            cls = (dev / "class").read_text().strip()
        except OSError:
            continue
        if vendor == "0x10de" and cls.startswith(("0x03", "0x12")):  # NVIDIA, видеоконтроллер / ускоритель
            return True
    return False


LINUX_DRIVER_HINT = """Видеокарта NVIDIA найдена, но драйвер не установлен (нет nvidia-smi). Установите драйвер и перезагрузитесь:
  Ubuntu:         sudo ubuntu-drivers install
  Debian:         sudo apt install nvidia-driver firmware-misc-nonfree   (нужен репозиторий non-free)
  Fedora:         sudo dnf install akmod-nvidia                          (нужен репозиторий RPM Fusion)
  Arch / Manjaro: sudo pacman -S nvidia        (или nvidia-open для видеокарт RTX 20xx и новее)
После перезагрузки команда nvidia-smi должна показать видеокарту. CUDA Toolkit ставить не нужно.
Если включена безопасная загрузка (Secure Boot), драйвер может потребовать подписи модуля (MOK)."""


def _run_py(python: Path | str, code: str) -> str | None:
    try:
        out = subprocess.run([str(python), "-c", code], capture_output=True, text=True, timeout=180)
        return out.stdout.strip() if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def torch_installed(python: Path | str) -> bool:
    return _run_py(python, "import torch") is not None


def torch_cuda_device(python: Path | str) -> str | None:
    """Название видеокарты, если PyTorch в этом окружении реально может считать на ней."""
    name = _run_py(python, "import torch\nif torch.cuda.is_available(): print(torch.cuda.get_device_name(0))")
    return name or None


def _pip(python: Path | str, *args: str) -> int:
    return subprocess.call([str(python), "-m", "pip", "--disable-pip-version-check", *args])


def install_cuda_torch(python: Path | str) -> bool:
    """Ставит PyTorch со сборкой CUDA, подходящей к драйверу, и проверяет, что видеокарта работает."""
    version = driver_cuda_version()
    builds = [b for b, need in CUDA_BUILDS if version is None or need <= version]
    if not builds:
        print(f"[gpu] Драйвер NVIDIA слишком старый (поддерживает CUDA {version[0]}.{version[1]}). "
              "Обновите драйвер — нужна поддержка CUDA 11.8 или новее.", flush=True)
        return False
    ver_text = f"CUDA до {version[0]}.{version[1]}" if version else "версию CUDA определить не удалось"
    print(f"[gpu] Видеокарта NVIDIA: драйвер поддерживает {ver_text}. Устанавливаю PyTorch с CUDA (~2–3 ГБ)…",
          flush=True)
    for build in builds:
        print(f"[gpu] Пробую сборку {build}…", flush=True)
        if torch_installed(python):
            _pip(python, "uninstall", "-y", "torch", "torchvision")
        if _pip(python, "install", "torch", "torchvision", "--index-url", TORCH_INDEX + build) == 0:
            name = torch_cuda_device(python)
            if name:
                print(f"[gpu] Готово: PyTorch ({build}) работает на видеокарте {name}.", flush=True)
                return True
        print(f"[gpu] Сборка {build} не заработала с этим драйвером — пробую следующую.", flush=True)
    print("[gpu] Не удалось включить видеокарту — возвращаю обычную сборку PyTorch (расчёт на процессоре).",
          flush=True)
    if torch_installed(python):
        _pip(python, "uninstall", "-y", "torch", "torchvision")
    _pip(python, "install", "torch", "torchvision")
    return False


def install_cuda_command() -> int:
    """python main.py --install-cuda — включить видеокарту NVIDIA в уже созданном окружении."""
    if sys.platform == "darwin":
        print("На macOS видеокарт NVIDIA нет: на Apple Silicon автоматически используется графика Metal (MPS).")
        return 0
    name = torch_cuda_device(sys.executable)
    if name:
        print(f"Видеокарта уже работает: {name}. Ничего делать не нужно.")
        return 0
    if nvidia_smi() is None:
        if linux_nvidia_hardware():
            print(LINUX_DRIVER_HINT)
        else:
            print("Видеокарта NVIDIA или её драйвер не найдены (нет nvidia-smi). "
                  "Установите драйвер с nvidia.com и запустите команду снова.")
        return 1
    return 0 if install_cuda_torch(sys.executable) else 1


def _pip_install(python: Path | str) -> None:
    print(f"[setup] Установка зависимостей ({REQUIREMENTS.name}) — это нужно только один раз…", flush=True)
    # Видеокарта NVIDIA: при первой установке сразу ставим PyTorch с подходящей сборкой CUDA
    if os.environ.get("AI_CAMERA_CPU_ONLY") != "1" and not torch_installed(python):
        if nvidia_smi() is not None:
            install_cuda_torch(python)
        elif linux_nvidia_hardware():
            print("[setup] " + LINUX_DRIVER_HINT + "\n[setup] Пока продолжаю без видеокарты.", flush=True)
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
