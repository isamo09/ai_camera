// Трансляция из браузера на сервер: камера этого устройства (ноутбук, телефон) или захват экрана/окна.
// Кадры рисуются в canvas, сжимаются в JPEG и отправляются POST-запросом на /api/push_frame.
const Broadcast = (() => {
  const MAX_WIDTH = 1280;          // крупнее для распознавания не нужно, а сеть разгружается
  let stream = null, video = null, canvas = null, ticker = null;
  let running = false, busy = false, mode = null, fps = 15;
  let sentTimes = [];
  let onChange = () => {};

  // Почему захват недоступен (или null, если всё в порядке)
  function unsupportedReason(m) {
    if (!window.isSecureContext) {
      return "Браузер даёт доступ к камере и экрану только по HTTPS или по адресу localhost / 127.0.0.1. " +
             "Для телефона запустите сервер так: main.py --https";
    }
    const md = navigator.mediaDevices;
    if (!md) return "Этот браузер не поддерживает захват видео.";
    if (m === "screen" && !md.getDisplayMedia) return "Этот браузер не умеет захватывать экран (на телефонах обычно недоступно).";
    if (m === "camera" && !md.getUserMedia) return "Этот браузер не поддерживает доступ к камере.";
    return null;
  }

  // Таймер в Web Worker не замедляется, когда вкладка в фоне (важно при захвате другого окна)
  function makeTicker() {
    const code = "let t=null;onmessage=e=>{clearInterval(t);if(e.data>0)t=setInterval(()=>postMessage(0),e.data)}";
    const w = new Worker(URL.createObjectURL(new Blob([code], { type: "text/javascript" })));
    w.onmessage = tick;
    return w;
  }

  async function acquire(m, { deviceId, width, height } = {}) {
    if (m === "screen") {
      return navigator.mediaDevices.getDisplayMedia({ video: { frameRate: { ideal: 30 } }, audio: false });
    }
    const video = { width: { ideal: width || 1280 }, height: { ideal: height || 720 } };
    if (deviceId) video.deviceId = { exact: deviceId };
    else video.facingMode = "environment"; // на телефоне — основная камера, на ноутбуке — любая
    return navigator.mediaDevices.getUserMedia({ video, audio: false });
  }

  // beforeSend — асинхронная функция, которую нужно выполнить до первой отправки (переключить источник на сервере)
  async function start(m, options = {}, beforeSend = null) {
    stop(false);
    const reason = unsupportedReason(m);
    if (reason) throw new Error(reason);

    stream = await acquire(m, options);           // сразу по клику — иначе браузер откажет
    video = document.createElement("video");
    video.muted = true;
    video.playsInline = true;
    video.srcObject = stream;
    await video.play();
    canvas = canvas || document.createElement("canvas");
    stream.getVideoTracks()[0].addEventListener("ended", () => stop()); // «Прекратить показ» в браузере

    if (beforeSend) await beforeSend();
    mode = m;
    running = true;
    sentTimes = [];
    ticker = ticker || makeTicker();
    ticker.postMessage(Math.round(1000 / fps));
    onChange();
  }

  async function tick() {
    if (!running || busy || !video || video.readyState < 2 || !video.videoWidth) return;
    busy = true;
    try {
      const scale = Math.min(1, MAX_WIDTH / video.videoWidth);
      canvas.width = Math.round(video.videoWidth * scale);
      canvas.height = Math.round(video.videoHeight * scale);
      canvas.getContext("2d").drawImage(video, 0, 0, canvas.width, canvas.height);
      const blob = await new Promise((resolve) => canvas.toBlob(resolve, "image/jpeg", 0.85));
      if (!blob) return;
      const res = await fetch("/api/push_frame", { method: "POST", body: blob, headers: { "Content-Type": "image/jpeg" } });
      if (res.status === 409) { stop(); return; }  // на сервере выбрали другой источник
      sentTimes.push(performance.now());
    } catch {
      // сеть моргнула — пропускаем кадр
    } finally {
      busy = false;
    }
  }

  function stop(notify = true) {
    const wasRunning = running;
    running = false;
    if (ticker) ticker.postMessage(0);
    if (stream) stream.getTracks().forEach((t) => t.stop());
    if (video) video.srcObject = null;
    stream = null;
    video = null;
    mode = null;
    if (notify && wasRunning) onChange();
  }

  function setFps(value) {
    fps = Math.max(1, Math.min(30, Number(value) || 15));
    if (running && ticker) ticker.postMessage(Math.round(1000 / fps));
  }

  function stats() {
    const now = performance.now();
    sentTimes = sentTimes.filter((t) => now - t < 2000);
    return {
      running, mode,
      fps: sentTimes.length / 2,
      size: video && video.videoWidth ? [video.videoWidth, video.videoHeight] : null,
    };
  }

  async function listCameras() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.enumerateDevices) return [];
    const devices = await navigator.mediaDevices.enumerateDevices();
    return devices.filter((d) => d.kind === "videoinput");
  }

  return {
    start, stop, setFps, stats, listCameras, unsupportedReason,
    set onChange(fn) { onChange = fn; },
    get running() { return running; },
    get mode() { return mode; },
  };
})();
