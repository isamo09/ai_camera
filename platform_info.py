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


_VIRTUAL_IFACE = ("vethernet", "vmware", "virtualbox", "vbox", "hyper-v", "wsl", "docker", "veth", "br-",
                  "virbr", "tun", "tap", "wg", "wireguard", "vpn", "amnezia", "zerotier", "tailscale", "utun",
                  "loopback", "bluetooth", "awdl", "llw", "bridge")


def _iface_rank(name: str) -> int | None:
    """Приоритет сетевого адаптера: Wi-Fi → Ethernet → прочие; None — виртуальный/служебный."""
    n = name.lower()
    if n in ("lo", "lo0"):
        return None
    if any(n.startswith(k) or f" {k}" in n or f"({k}" in n or k in n.split() for k in _VIRTUAL_IFACE) \
            or any(k in n for k in ("vmware", "virtualbox", "vethernet", "hyper-v", "wireguard", "openvpn", "amnezia")):
        return None
    if any(k in n for k in ("wi-fi", "wifi", "wlan", "wireless", "беспровод")) or n.startswith(("wl", "en0")):
        return 0
    if any(k in n for k in ("ethernet", "локальн", "local area")) or n.startswith(("eth", "en")):
        return 1
    return 2


def lan_ips() -> list[str]:
    """IP-адреса компьютера в локальной сети (для телефона), лучший — первый. VPN и виртуальные адаптеры пропускаются."""
    import ipaddress

    found: list[tuple[int, str]] = []
    try:
        import psutil

        stats = psutil.net_if_stats()
        for name, addrs in psutil.net_if_addrs().items():
            rank = _iface_rank(name)
            if rank is None or (name in stats and not stats[name].isup):
                continue
            for a in addrs:
                if a.family != socket.AF_INET:
                    continue
                ip = ipaddress.ip_address(a.address)
                if ip.is_private and not ip.is_loopback and not ip.is_link_local:
                    found.append((rank, a.address))
    except Exception:  # noqa: BLE001 — psutil нет: берём адрес маршрута по умолчанию
        pass
    ips = [ip for _, ip in sorted(found)]
    if not ips:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.connect(("10.255.255.255", 1))  # пакет никуда не отправляется
                ips.append(s.getsockname()[0])
        except OSError:
            pass
    return ips


_VPN_IFACE = ("vpn", "amnezia", "wireguard", "wg", "openvpn", "tun", "tap", "tailscale", "zerotier", "utun", "ppp")


def vpn_ips() -> list[str]:
    """Адреса VPN-адаптеров (WireGuard, AmneziaVPN, OpenVPN, Tailscale, ZeroTier…) — по ним страница
    открывается с устройств, подключённых к той же VPN-сети."""
    import ipaddress

    cgnat = ipaddress.ip_network("100.64.0.0/10")  # Tailscale и другие оверлей-сети
    ips = []
    try:
        import psutil

        stats = psutil.net_if_stats()
        for name, addrs in psutil.net_if_addrs().items():
            n = name.lower()
            if not any(k in n for k in _VPN_IFACE) or (name in stats and not stats[name].isup):
                continue
            for a in addrs:
                if a.family != socket.AF_INET:
                    continue
                ip = ipaddress.ip_address(a.address)
                if (ip.is_private or ip in cgnat) and not ip.is_link_local and not ip.is_loopback:
                    ips.append(a.address)
    except Exception:  # noqa: BLE001
        pass
    return ips


def lan_ip() -> str | None:
    ips = lan_ips()
    return ips[0] if ips else None


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

def _torch():
    try:
        import torch
        return torch
    except ImportError:
        return None


def _mps_available(torch) -> bool:
    mps = getattr(torch.backends, "mps", None) if torch else None
    return bool(mps is not None and mps.is_available())


def pick_device(preferred: str = "auto") -> str:
    """Возвращает реально доступное устройство: cuda (NVIDIA) → mps (Apple Silicon) → cpu.

    Если запрошено недоступное устройство (например, cuda без видеокарты), выбирается лучшее доступное.
    """
    torch = _torch()
    if torch is None:
        return "cpu"
    preferred = (preferred or "auto").strip()
    if preferred == "cpu":
        return "cpu"
    if preferred.startswith("cuda") and torch.cuda.is_available():
        idx = int(preferred.split(":")[1]) if ":" in preferred else 0
        return f"cuda:{idx}" if idx < torch.cuda.device_count() else "cuda:0"
    if preferred == "mps" and _mps_available(torch):
        return "mps"
    # auto или запрошенное устройство недоступно
    if torch.cuda.is_available():
        return "cuda:0"
    if _mps_available(torch):
        return "mps"
    return "cpu"


def nvidia_gpus_from_driver() -> list[str]:
    """Видеокарты NVIDIA по данным драйвера (даже если PyTorch собран без CUDA)."""
    from bootstrap import nvidia_smi

    out = nvidia_smi("--query-gpu=name", "--format=csv,noheader")
    return [ln.strip() for ln in (out or "").splitlines() if ln.strip()]


_devices_cache: list[dict] | None = None


def list_devices() -> list[dict]:
    """Все вычислительные устройства компьютера для выбора в интерфейсе."""
    global _devices_cache
    if _devices_cache is not None:
        return _devices_cache
    torch = _torch()
    best = pick_device("auto")
    devices = [{"id": "auto", "title": f"Авто — лучшее доступное: {device_title(best)}", "available": True}]
    devices.append({"id": "cpu", "title": f"Процессор: {cpu_name()}", "available": True})
    cuda_count = torch.cuda.device_count() if torch is not None and torch.cuda.is_available() else 0
    for i in range(cuda_count):
        props = torch.cuda.get_device_properties(i)
        devices.append({"id": f"cuda:{i}", "available": True,
                        "title": f"Видеокарта NVIDIA: {props.name} ({props.total_memory / 1024**3:.0f} ГБ)"})
    if cuda_count == 0:
        from bootstrap import linux_nvidia_hardware

        names = nvidia_gpus_from_driver()
        for name in names:
            devices.append({"id": "cuda:0", "available": False,
                            "title": f"Видеокарта NVIDIA: {name} — недоступна, установлен PyTorch без CUDA",
                            "fix": "Остановите программу и выполните: python main.py --install-cuda"})
        if not names and linux_nvidia_hardware():
            devices.append({"id": "cuda:0", "available": False,
                            "title": "Видеокарта NVIDIA — недоступна, не установлен драйвер",
                            "fix": "Установите драйвер NVIDIA (например, sudo ubuntu-drivers install), "
                                   "перезагрузитесь и выполните: python main.py --install-cuda"})
    if torch is not None and _mps_available(torch):
        devices.append({"id": "mps", "title": "Графика Apple Silicon (Metal)", "available": True})
    _devices_cache = devices
    return devices


# ---------------------------------------------------------------- поиск камер

def _preferred_enum_backend():
    import cv2

    if WINDOWS:
        return cv2.CAP_DSHOW  # тот же порядок номеров, что и при открытии камеры
    if MACOS:
        return getattr(cv2, "CAP_AVFOUNDATION", 1200)
    return cv2.CAP_V4L2


def list_cameras(skip_probe: set[int] | None = None) -> list[dict]:
    """Камеры, подключённые к компьютеру: [{"index": 0, "name": "USB CAMERA"}, ...]."""
    try:
        from cv2_enumerate_cameras import enumerate_cameras

        seen, cams = set(), []
        for c in enumerate_cameras(_preferred_enum_backend()):
            if c.index in seen:
                continue
            seen.add(c.index)
            cams.append({"index": c.index, "name": c.name or f"Камера {c.index}"})
        if cams or not LINUX:
            return cams
    except Exception:  # noqa: BLE001 — пакета нет или ОС не поддерживается
        pass

    if LINUX:  # имена устройств из sysfs
        cams = []
        for dev in sorted(Path("/sys/class/video4linux").glob("video*")):
            try:
                idx = int(dev.name.replace("video", ""))
                cams.append({"index": idx, "name": (dev / "name").read_text().strip()})
            except (OSError, ValueError):
                continue
        if cams:
            return cams

    # Последний вариант — пробуем открыть первые номера (занятую нами камеру не трогаем)
    import cv2

    cams = []
    for idx in range(4):
        if skip_probe and idx in skip_probe:
            cams.append({"index": idx, "name": f"Камера {idx}"})
            continue
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            cams.append({"index": idx, "name": f"Камера {idx}"})
        cap.release()
    return cams


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
