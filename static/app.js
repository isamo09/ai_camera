// Страница камеры: живые данные, настройки и снимки
const PEOPLE_ANIMALS = ["person", "bird", "cat", "dog", "horse", "sheep", "cow",
                        "elephant", "bear", "zebra", "giraffe", "teddy bear"];

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
  $("#sourceInput").value = settings.camera_source;
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

  const applySource = () => {
    const v = $("#sourceInput").value.trim();
    if (v) { saveSettings({ camera_source: v }, 0); toast(`Источник: <b>${escapeHtml(v)}</b>`); }
  };
  $("#sourceApply").addEventListener("click", applySource);
  $("#sourceInput").addEventListener("keydown", (e) => e.key === "Enter" && applySource());

  $("#resolution").addEventListener("change", (e) => {
    const [width, height] = e.target.value.split("x").map(Number);
    saveSettings({ width, height }, 0);
  });
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
  $("#classesPeople").addEventListener("click", () => setClasses(PEOPLE_ANIMALS));
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

function renderStatus(s) {
  const pill = $("#livePill");
  const texts = { running: "В эфире", opening: "Подключение к камере…",
                  loading_model: "Загрузка модели…", error: "Ошибка", starting: "Запуск…" };
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
  const models = await api("/api/models");
  $("#modelSelect").innerHTML = models.map((m) => `<option value="${m.id}">${escapeHtml(m.title)}</option>`).join("");
  settings = await api("/api/settings");
  await loadClasses();
  applySettingsToForm();
  bindSettings();
  bindClasses();

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
