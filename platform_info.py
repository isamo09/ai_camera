"""Всё, что зависит от операционной системы: камера, шрифты, порт, вычислительное устройство."""
from __future__ import annotations

import os
import platform
import socket
import sys
from pathlib import Path

WINDOWS = sys.platform.startswith("win")
MACOS = sys.platform == "darwin"
LINUX = sys.platform.startswith("linux")
OS_NAME = "Windows" if WINDOWS else "macOS" if MACOS else "Linux" if LINUX else platform.system()


def os_description() -> str:
    if MACOS:
        return f"macOS {platform.mac_ver()[0]} ({platform.machine()})"
    return f"{OS_NAME} {platform.release()} ({platform.machine()})"


# ---------------------------------------------------------------- веб-сервер

def default_port() -> int:
    # На macOS 12+ порт 5000 занят «Приёмником AirPlay»
    return 8080 if MACOS else 5000


def port_is_free(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(("" if host == "0.0.0.0" else host, port))
            return True
        except OSError:
            return False


def find_free_port(host: str, port: int, attempts: int = 30) -> int:
    for p in range(port, port + attempts):
        if port_is_free(host, p):
            return p
    raise RuntimeError(f"Не найден свободный порт в диапазоне {port}–{port + attempts - 1}")


def lan_ip() -> str | None:
    """IP-адрес компьютера в локальной сети (чтобы открыть страницу с телефона)."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("10.255.255.255", 1))  # пакет никуда не отправляется
            return s.getsockname()[0]
    except OSError:
        return None


def can_open_browser() -> bool:
    if LINUX:
        return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    return True


# ---------------------------------------------------------------- камера

def camera_backends() -> list[tuple[int, str]]:
    """Способы доступа к камере в порядке предпочтения для текущей ОС."""
    import cv2

    if WINDOWS:
        order = [("CAP_DSHOW", "DirectShow"), ("CAP_MSMF", "Media Foundation")]
    elif MACOS:
        order = [("CAP_AVFOUNDATION", "AVFoundation")]
    else:
        order = [("CAP_V4L2", "Video4Linux2")]
    backends = [(getattr(cv2, name), title) for name, title in order if hasattr(cv2, name)]
    backends.append((cv2.CAP_ANY, "авто"))
    return backends


def camera_hint() -> str:
    """Подсказка, что делать, если камера не открывается или не отдаёт кадры."""
    if WINDOWS:
        return ("Закройте программы, которые могут занимать камеру (Камера Windows, Zoom, Teams, Discord).\n"
                "Параметры → Конфиденциальность → Камера → разрешите доступ классическим приложениям.")
    if MACOS:
        return ("Разрешите доступ: Системные настройки → Конфиденциальность и безопасность → Камера →\n"
                "включите Терминал (или ваш редактор кода) и перезапустите программу.\n"
                "Если рядом iPhone, он может занять номер 0 — попробуйте источник 1.")
    return ("Проверьте, что камера видна в системе (ls /dev/video*) и у пользователя есть права\n"
            "(sudo usermod -aG video $USER, затем перелогиньтесь). Закройте другие программы с камерой.")


# ---------------------------------------------------------------- шрифты с кириллицей

def font_candidates() -> list[str]:
    windir = os.environ.get("WINDIR", "C:/Windows")
    win = [str(Path(windir) / "Fonts" / n) for n in ("segoeuib.ttf", "arialbd.ttf", "arial.ttf", "tahomabd.ttf")]
    mac = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial Unicode.ttf",
        "/System/Library/Fonts/Supplemental/Verdana Bold.ttf",
    ]
    linux = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",   # Debian / Ubuntu
        "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans-Bold.ttf",  # Fedora
        "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",                # Arch
        "/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf",
        "/usr/share/fonts/noto/NotoSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    first = win if WINDOWS else mac if MACOS else linux
    return first + [p for p in win + mac + linux if p not in first]


def find_font() -> str | None:
    for path in font_candidates():
        if Path(path).exists():
            return path
    return None


# ---------------------------------------------------------------- вычисления

def pick_device(preferred: str = "auto") -> str:
    """cuda (NVIDIA) → mps (Apple Silicon) → cpu."""
    if preferred and preferred != "auto":
        return preferred
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda:0"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def device_title(device: str) -> str:
    if device.startswith("cuda"):
        try:
            import torch
            return f"NVIDIA GPU ({torch.cuda.get_device_name(0)})"
        except Exception:  # noqa: BLE001
            return "NVIDIA GPU"
    if device == "mps":
        return "Apple GPU (Metal)"
    return f"CPU ({cpu_name()})"


def cpu_name() -> str:
    """Понятное название процессора на любой ОС."""
    try:
        if WINDOWS:
            import winreg

            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            return winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
        if MACOS:
            import subprocess

            out = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True)
            if out.stdout.strip():
                return out.stdout.strip()
        if LINUX:
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.lower().startswith(("model name", "hardware")):
                    return line.split(":", 1)[1].strip()
    except Exception:  # noqa: BLE001
        pass
    return platform.processor() or platform.machine()


def model_download_hint() -> str:
    if MACOS:
        return ("Нет доступа к интернету или не установлены сертификаты Python. Для Python с python.org\n"
                "запустите «Install Certificates.command» из папки /Applications/Python 3.x/.\n"
                "Либо скачайте файл модели вручную и положите его в папку models/.")
    return "Проверьте подключение к интернету или скачайте файл модели вручную в папку models/."
