// Общие помощники для всех страниц
const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];

async function api(url, options = {}) {
  const opts = { ...options };
  if (opts.body && typeof opts.body !== "string") {
    opts.body = JSON.stringify(opts.body);
    opts.headers = { "Content-Type": "application/json", ...(opts.headers || {}) };
  }
  const res = await fetch(url, opts);
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.error || `Ошибка ${res.status}`);
  return data;
}

let toastTimer;
function toast(html, type = "info", ms = 3500) {
  const el = $("#toast");
  el.innerHTML = html;
  el.className = `toast show ${type}`;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => (el.className = "toast"), ms);
}

function setGalleryCount(n) {
  const b = $("#galleryCount");
  if (b) b.textContent = n > 0 ? n : "";
}

function escapeHtml(s) {
  return String(s ?? "").replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
}

function formatDate(iso) {
  const d = new Date(iso);
  return d.toLocaleString("ru-RU", { day: "2-digit", month: "long", year: "numeric",
                                     hour: "2-digit", minute: "2-digit", second: "2-digit" });
}

function chipsHtml(counts) {
  return counts.map((c) =>
    `<span class="chip"><i style="background:${c.color}"></i>${escapeHtml(c.label)} × ${c.count}</span>`
  ).join("");
}
