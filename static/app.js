// Страница камеры: живые данные, настройки и снимки
const PEOPLE_ANIMALS = ["person", "man", "woman", "boy", "girl", "human face", "animal", "mammal", "bird", "cat",
                        "dog", "horse", "sheep", "cow", "cattle", "elephant", "bear", "zebra", "giraffe", "teddy bear"];

let settings = {};
let classes = [];
let selectedClasses = new Set();

// ------------------------------------------------------------ настройки
function writeInput(el, value) {
  if (el.type === "checkbox") el.checked = !!value;
  else el.value = value;
  updateOutput(el);
}

function readInput(el) {
  if (el.type === "checkbox") return el.checked;
  if (el.type === "range" || el.dataset.key === "imgsz") return Number(el.value);
  return el.value;
}

function updateOutput(el) {
  const out = $(`output[data-for="${el.dataset.key}"]`);
  if (!out) return;
  const v = Number(el.value);
  out.textContent = ["confidence", "iou"].includes(el.dataset.key) ? `${Math.round(v * 100)}%` : v;
}

const pending = {};
let saveTimer;
function saveSettings(patch, delay = 250) {
  Object.assign(pending, patch);
  clearTimeout(saveTimer);
  saveTimer = setTimeout(async () => {
    const body = { ...pending };
    for (const k in pending) delete pending[k];
    try {
      const before = settings.language;
      settings = await api("/api/settings", { method: "POST", body });
      if (settings.language !== before) await loadClasses();
    } catch (e) {
      toast(`Не удалось сохранить настройки: ${escapeHtml(e.message)}`, "error");
    }
  }, delay);
}

function applySettingsToForm() {
  $$("[data-key]").forEach((el) => {
    if (el.dataset.key in settings) writeInput(el, settings[el.dataset.key]);
  });
  if (!isDeviceSource(settings.camera_source) && settings.camera_source !== "browser") {
    $("#sourceInput").value = settings.camera_source;
  }
  renderModelInfo();
  const res = `${settings.width}x${settings.height}`;
  const sel = $("#resolution");
  if (![...sel.options].some((o) => o.value === res)) sel.add(new Option(res.replace("x", " × "), res));
  sel.value = res;
  selectedClasses = new Set(settings.classes);
  renderClasses();
}

function bindSettings() {
  $$("[data-key]").forEach((el) => {
    const handler = () => {
      updateOutput(el);
      saveSettings({ [el.dataset.key]: readInput(el) });
    };
    el.addEventListener(el.type === "range" ? "input" : "change", handler);
  });

  $("#resolution").addEventListener("change", (e) => {
    const [width, height] = e.target.value.split("x").map(Number);
    saveSettings({ width, height }, 0);
  });
  $("#modelSelect").addEventListener("change", renderModelInfo);
}

// ------------------------------------------------------------ модели и устройства
let models = [];
let devices = [];

async function loadModels() {
  models = await api("/api/models");
  const families = [...new Set(models.map((m) => m.family))];
  $("#modelSelect").innerHTML = families.map((f) => `<optgroup label="${escapeHtml(f)}">` +
    models.filter((m) => m.family === f).map((m) => {
      const stats = m.map ? ` — ${m.map} mAP · ${m.params} млн пар.` : "";
      return `<option value="${escapeHtml(m.id)}">${escapeHtml(m.title)}${stats}${m.downloaded ? "" : " ⬇"}</option>`;
    }).join("") + "</optgroup>").join("");
  if (settings.model) $("#modelSelect").value = settings.model;
  renderModelInfo();
}

function renderModelInfo() {
  const m = models.find((x) => x.id === $("#modelSelect").value);
  const box = $("#modelInfo");
  if (!m) { box.innerHTML = ""; return; }
  if (m.custom) {
    box.innerHTML = `<p class="muted small">Собственная модель из папки <code>models/</code>. Классы берутся из неё.</p>`;
    return;
  }
  const row = (label, value, pct) => `<div class="mi-row"><span>${label}</span><b>${value}</b></div>` +
    (pct !== undefined ? `<div class="mi-bar"><span style="width:${pct}%"></span></div>` : "");
  box.innerHTML =
    row("Классов", m.open_vocab ? "любые — свои названия" : `${m.classes} (${m.dataset})`) +
    row(`Точность (mAP50-95, ${m.dataset})`, `${m.map}%`, Math.round((m.map / 60) * 100)) +
    row("Параметров", `${m.params} млн`) +
    row("Вычислений на кадр", `${m.flops} млрд`) +
    (m.cpu_ms ? row("Скорость на CPU (офиц.)", `≈ ${Math.round(m.cpu_ms)} мс/кадр`,
                    Math.round(Math.min(100, (m.cpu_ms / 860) * 100))) : "") +
    `<p class="muted small">${escapeHtml(m.hint)}. ${m.downloaded ? "Файл уже скачан." : "Будет скачана при выборе."}` +
    (m.dataset !== "COCO" ? " Точность измерена на другом наборе данных, поэтому с моделями COCO напрямую не сравнивается." : "") +
    `</p>`;
  renderVocab();
}

// ------------------------------------------------------------ свои названия (YOLOE)
let vocab = { items: [], applied: [] };
const selectedModel = () => models.find((x) => x.id === $("#modelSelect").value);

async function loadVocab() {
  try { vocab = await api("/api/vocab"); } catch { /* сервер перезапускается */ }
  renderVocab();
}

function renderVocab() {
  const m = selectedModel();
  const card = $("#vocabCard");
  if (!card) return;
  card.hidden = !(m && m.open_vocab);
  if (card.hidden) return;
  const applied = new Set(vocab.applied || []);
  $("#vocabList").innerHTML = (vocab.items || []).map((it, i) => {
    const warn = !it.known ? ` <span class="vocab-warn" title="Перевод не найден — укажите: ${escapeHtml(it.label)} = english">⚠</span>` : "";
    const state = applied.has(it.prompt) ? "" : " pending";
    return `<span class="chip vocab-chip${state}" title="Запрос к модели: ${escapeHtml(it.prompt)}">` +
      `${escapeHtml(it.label)}${it.prompt !== it.label.toLowerCase() ? ` <i>→ ${escapeHtml(it.prompt)}</i>` : ""}${warn}` +
      `<button data-i="${i}" aria-label="Удалить">×</button></span>`;
  }).join("") || `<span class="muted small">Список пуст — добавьте названия, которые нужно искать.</span>`;
}

async function saveVocab(list) {
  settings = await api("/api/settings", { method: "POST", body: { custom_classes: list } });
  await loadVocab();
}

function bindVocab() {
  const add = async () => {
    const input = $("#vocabInput");
    const parts = input.value.split(/[,;\n]/).map((x) => x.trim()).filter(Boolean);
    if (!parts.length) return;
    input.value = "";
    await saveVocab([...settings.custom_classes, ...parts]);
    const unknown = vocab.items.filter((it) => !it.known && parts.includes(it.entry));
    if (unknown.length) toast(`Нет перевода для: ${unknown.map((u) => escapeHtml(u.entry)).join(", ")}. ` +
      `Модель понимает английский — напишите, например, «${escapeHtml(unknown[0].entry)} = …»`, "error", 8000);
  };
  $("#vocabAdd").addEventListener("click", add);
  $("#vocabInput").addEventListener("keydown", (e) => e.key === "Enter" && add());
  $("#vocabList").addEventListener("click", (e) => {
    const i = e.target.dataset.i;
    if (i === undefined) return;
    const list = [...settings.custom_classes];
    list.splice(Number(i), 1);
    saveVocab(list);
  });
  $("#vocabClear").addEventListener("click", () => saveVocab([]));
}

async function loadDevices() {
  devices = await api("/api/devices");
  $("#deviceSelect").innerHTML = devices.map((d) =>
    `<option value="${escapeHtml(d.id)}" ${d.available ? "" : "disabled"}>${escapeHtml(d.title)}</option>`).join("");
  $("#deviceSelect").value = settings.device;
}

// ------------------------------------------------------------ источник видео
let hostCameras = [];
const isDeviceSource = (src) => /^\d+$/.test(String(src));
const browserMode = () => { try { return localStorage.getItem("browserMode") || "camera"; } catch { return "camera"; } };
const setBrowserMode = (m) => { try { localStorage.setItem("browserMode", m); } catch { /* приватный режим */ } };

async function loadHostCameras(refresh = false) {
  try {
    hostCameras = await api("/api/cameras" + (refresh ? "?refresh=1" : ""));
  } catch { hostCameras = []; }
  renderSourceSelect();
  if (refresh) toast(hostCameras.length ? `Найдено камер: ${hostCameras.length}` : "Камеры на компьютере не найдены");
}

function currentSourceValue() {
  const src = settings.camera_source;
  if (src === "browser") return `browser-${browserMode()}`;
  if (isDeviceSource(src)) return `cam:${src}`;
  return "url";
}

function renderSourceSelect() {
  const cams = [...hostCameras];
  if (isDeviceSource(settings.camera_source) && !cams.some((c) => String(c.index) === settings.camera_source)) {
    cams.push({ index: Number(settings.camera_source), name: `Камера ${settings.camera_source}` });
  }
  const opt = (v, t, extra = "") => `<option value="${v}" ${extra}>${escapeHtml(t)}</option>`;
  $("#sourceSelect").innerHTML =
    `<optgroup label="Камеры компьютера, где запущен сервер">` +
    (cams.length ? cams.map((c) => opt(`cam:${c.index}`, `📷 ${c.name} (№${c.index})`)).join("")
                 : opt("", "Камеры не найдены", "disabled")) +
    `</optgroup><optgroup label="Через браузер (это устройство)">` +
    opt("browser-camera", "🌐 Камера этого устройства") +
    opt("browser-screen", "🖥 Экран или окно (захват экрана)") +
    `</optgroup><optgroup label="Другое">` +
    opt("url", "🔗 IP-камера, поток или видеофайл…") + `</optgroup>`;
  $("#sourceSelect").value = currentSourceValue();
  updateSourceBlocks();
}

function updateSourceBlocks() {
  const v = $("#sourceSelect").value;
  $("#urlBlock").hidden = v !== "url";
  $("#browserBlock").hidden = !v.startsWith("browser");
  $("#browserCamField").hidden = v !== "browser-camera";
  $("#resolutionField").hidden = v === "browser-screen" || v === "url";
  renderBroadcastState();
}

async function refreshBrowserCameras() {
  const cams = await Broadcast.listCameras().catch(() => []);
  const sel = $("#browserCamSelect");
  const current = sel.value;
  sel.innerHTML = `<option value="">По умолчанию (основная камера)</option>` +
    cams.map((c, i) => `<option value="${escapeHtml(c.deviceId)}">${escapeHtml(c.label || `Камера ${i + 1}`)}</option>`).join("");
  sel.value = [...sel.options].some((o) => o.value === current) ? current : "";
}

async function startBroadcast(mode) {
  setBrowserMode(mode);
  try {
    await Broadcast.start(mode,
      { deviceId: $("#browserCamSelect").value, width: settings.width, height: settings.height },
      async () => { settings = await api("/api/settings", { method: "POST", body: { camera_source: "browser" } }); });
    toast(mode === "screen" ? "Трансляция экрана началась" : "Трансляция камеры началась", "success");
    if (mode === "camera") refreshBrowserCameras();  // названия камер доступны после разрешения
  } catch (e) {
    const msg = e.name === "NotAllowedError" ? "Доступ запрещён — разрешите камеру или показ экрана в браузере."
      : e.name === "NotFoundError" ? "Камера на этом устройстве не найдена."
      : e.name === "NotReadableError" ? "Камера занята другой программой."
      : e.message;
    toast(escapeHtml(msg), "error", 7000);
  }
  renderBroadcastState();
}

function renderBroadcastState() {
  const v = $("#sourceSelect").value;
  if (!v || !v.startsWith("browser")) return;
  const mode = v.slice("browser-".length);
  const st = Broadcast.stats();
  const btn = $("#broadcastBtn");
  const reason = Broadcast.unsupportedReason(mode);
  btn.disabled = !!reason;
  btn.textContent = st.running ? "■ Остановить трансляцию" : "▶ Начать трансляцию";
  btn.classList.toggle("danger", st.running);
  btn.classList.toggle("primary", !st.running);
  $("#broadcastInfo").textContent = reason ? reason
    : st.running ? `Идёт трансляция ${st.mode === "screen" ? "экрана" : "камеры"}` +
                   (st.size ? ` ${st.size[0]}×${st.size[1]}` : "") + `, отправляется ${st.fps.toFixed(1)} кадр/с`
    : settings.camera_source === "browser"
      ? "Сервер ждёт кадры. Нажмите «Начать трансляцию»."
      : "Кадры с этого устройства будут отправляться на сервер для распознавания.";
}

function bindSource() {
  $("#sourceSelect").addEventListener("change", (e) => {
    const v = e.target.value;
    if (v.startsWith("cam:")) {
      Broadcast.stop();
      saveSettings({ camera_source: v.slice(4) }, 0);
    } else if (v === "url") {
      Broadcast.stop();
      $("#sourceInput").focus();
    } else {
      startBroadcast(v.slice("browser-".length)); // сразу, пока действует клик пользователя
    }
    updateSourceBlocks();
  });

  const applyUrl = () => {
    const v = $("#sourceInput").value.trim();
    if (v) { saveSettings({ camera_source: v }, 0); toast(`Источник: <b>${escapeHtml(v)}</b>`); }
  };
  $("#sourceApply").addEventListener("click", applyUrl);
  $("#sourceInput").addEventListener("keydown", (e) => e.key === "Enter" && applyUrl());

  $("#camerasRefresh").addEventListener("click", () => loadHostCameras(true));
  $("#broadcastBtn").addEventListener("click", () => {
    if (Broadcast.running) Broadcast.stop();
    else startBroadcast($("#sourceSelect").value.slice("browser-".length));
  });
  $("#browserCamSelect").addEventListener("change", () => {
    if (Broadcast.running && Broadcast.mode === "camera") startBroadcast("camera");
  });
  $("#pushFps").addEventListener("input", (e) => {
    $("#pushFpsOut").textContent = e.target.value;
    Broadcast.setFps(e.target.value);
  });
  Broadcast.onChange = renderBroadcastState;
}

// ------------------------------------------------------------ классы
async function loadClasses() {
  classes = await api("/api/classes");
  renderClasses();
}

function renderClasses() {
  const q = $("#classSearch").value.trim().toLowerCase();
  const list = $("#classList");
  list.innerHTML = classes
    .filter((c) => !q || c.label.toLowerCase().includes(q) || c.name.includes(q))
    .map((c) => `<label class="class-item"><input type="checkbox" value="${escapeHtml(c.name)}"
        ${selectedClasses.has(c.name) ? "checked" : ""}><span>${escapeHtml(c.label)}</span></label>`)
    .join("");
  const n = selectedClasses.size;
  $("#classesBadge").textContent = n ? n : "все";
}

function setClasses(names) {
  selectedClasses = new Set(names);
  renderClasses();
  saveSettings({ classes: [...selectedClasses] });
}

function bindClasses() {
  $("#classSearch").addEventListener("input", renderClasses);
  $("#classList").addEventListener("change", (e) => {
    if (e.target.checked) selectedClasses.add(e.target.value);
    else selectedClasses.delete(e.target.value);
    setClasses(selectedClasses);
  });
  $("#classesAll").addEventListener("click", () => setClasses(classes.map((c) => c.name)));
  $("#classesNone").addEventListener("click", () => setClasses([]));
  $("#classesPeople").addEventListener("click", () =>
    setClasses(classes.filter((c) => PEOPLE_ANIMALS.includes(c.name.toLowerCase())).map((c) => c.name)));
}

// ------------------------------------------------------------ живой статус
async function pollStatus() {
  try {
    const s = await api("/api/status");
    renderStatus(s);
  } catch {
    $("#liveText").textContent = "Сервер недоступен";
    $("#livePill").className = "live-pill error";
  } finally {
    setTimeout(pollStatus, 400);
  }
}

let lastModel = null;
let lastVocab = null;

function renderStatus(s) {
  const pill = $("#livePill");
  const texts = { running: "В эфире", opening: "Подключение к камере…", loading_model: "Загрузка модели…",
                  error: "Ошибка", starting: "Запуск…", waiting_browser: "Ожидание трансляции из браузера" };

  // источник или модель могли поменять в другой вкладке — синхронизируем
  if (s.source !== settings.camera_source) {
    settings.camera_source = s.source;
    if (s.source !== "browser" && Broadcast.running) Broadcast.stop();
    renderSourceSelect();
  }
  if (s.model && s.model !== lastModel) {
    if (lastModel) loadModels();  // обновить отметки «скачана»
    lastModel = s.model;
    loadClasses();               // у каждой модели свой список классов
    loadVocab();
  }
  const vocabKey = (s.vocab || []).join("|");
  if (vocabKey !== lastVocab) {  // свои названия применены — обновить классы и отметки
    lastVocab = vocabKey;
    loadClasses();
    loadVocab();
  }
  $("#deviceInfo").textContent = `Сейчас используется: ${s.device}` +
    (devices.some((d) => d.fix) ? `. ${devices.find((d) => d.fix).fix}` : "");
  renderBroadcastState();
  pill.className = `live-pill ${s.status}`;
  $("#liveText").textContent = s.error && s.status !== "running" ? s.error : (texts[s.status] || s.status);

  $("#statTotal").textContent = s.total;
  $("#statKinds").textContent = s.counts.length;
  $("#statFps").textContent = s.fps;
  setGalleryCount(s.gallery_count);
  if (s.frame_size[0]) $("#frameInfo").textContent =
    `Кадр: ${s.frame_size[0]} × ${s.frame_size[1]} · модель: ${s.model || "—"} · ` +
    `вычисления: ${s.device} · ${s.os}, камера через ${s.backend || "—"}`;

  $("#counts").innerHTML = s.counts.length
    ? s.counts.map((c) => `<li><i style="background:${c.color}"></i><span>${escapeHtml(c.label)}</span><b>${c.count}</b></li>`).join("")
    : `<li class="muted">Ничего не найдено</li>`;

  $("#objectsHint").textContent = s.objects.length ? `${s.objects.length} шт.` : "";
  $("#objects").innerHTML = s.objects.length
    ? s.objects.map((o) => `
        <div class="object" style="--c:${o.color}">
          <span class="object-name">${escapeHtml(o.label)}</span>
          <span class="object-pos">📍 ${escapeHtml(o.position_text)}</span>
          <span class="object-conf"><span style="width:${Math.round(o.conf * 100)}%"></span></span>
          <span class="object-pct">${Math.round(o.conf * 100)}%</span>
        </div>`).join("")
    : `<p class="muted">${settings.detect_enabled === false ? "Распознавание выключено" : "Объекты не обнаружены"}</p>`;
}

// ------------------------------------------------------------ снимок
async function takeSnapshot() {
  const btn = $("#snapBtn");
  btn.disabled = true;
  try {
    const meta = await api("/api/snapshot", { method: "POST", body: { note: $("#note").value } });
    const flash = $("#flash");
    flash.classList.remove("go"); void flash.offsetWidth; flash.classList.add("go");
    $("#note").value = "";
    toast(`Снимок сохранён: ${meta.total} объект(ов). <a href="/gallery">Открыть галерею →</a>`, "success");
  } catch (e) {
    toast(escapeHtml(e.message), "error");
  } finally {
    btn.disabled = false;
  }
}

// ------------------------------------------------------------ старт
async function init() {
  settings = await api("/api/settings");
  await Promise.all([loadModels(), loadDevices(), loadClasses()]);
  applySettingsToForm();
  bindSettings();
  bindClasses();
  bindSource();
  bindVocab();
  loadVocab();
  loadHostCameras();

  $("#snapBtn").addEventListener("click", takeSnapshot);
  document.addEventListener("keydown", (e) => {
    if ((e.key === "s" || e.key === "ы") && !["INPUT", "TEXTAREA", "SELECT"].includes(document.activeElement.tagName)) {
      e.preventDefault();
      takeSnapshot();
    }
  });
  // если поток оборвался (например, перезапуск сервера) — переподключаемся
  $("#video").addEventListener("error", () => setTimeout(() => ($("#video").src = `/video_feed?t=${Date.now()}`), 1500));
  pollStatus();
}

init().catch((e) => toast(`Ошибка инициализации: ${escapeHtml(e.message)}`, "error", 8000));
