// Страница галереи
let items = [];
let current = null;
let showRaw = false;

async function loadGallery() {
  items = await api("/api/gallery");
  setGalleryCount(items.length);
  render();
}

function matches(item, q) {
  if (!q) return true;
  const hay = [item.note, item.description, ...item.counts.map((c) => `${c.label} ${c.name}`)].join(" ").toLowerCase();
  return hay.includes(q);
}

function render() {
  const q = $("#gallerySearch").value.trim().toLowerCase();
  const list = items.filter((i) => matches(i, q));
  $("#empty").hidden = items.length > 0;
  $("#grid").innerHTML = list.map((i) => `
    <article class="shot" data-id="${i.id}">
      <div class="shot-img"><img loading="lazy" src="/media/${i.image}" alt="Снимок ${escapeHtml(i.created)}">
        <span class="shot-total">${i.total}</span></div>
      <div class="shot-body">
        <div class="shot-date">${formatDate(i.created)}</div>
        ${i.note ? `<div class="shot-note">${escapeHtml(i.note)}</div>` : ""}
        <div class="chips">${chipsHtml(i.counts) || '<span class="muted small">Объектов нет</span>'}</div>
      </div>
    </article>`).join("") || (items.length ? `<p class="empty">Ничего не найдено</p>` : "");
}

function openItem(id) {
  current = items.find((i) => i.id === id);
  if (!current) return;
  showRaw = false;
  updateLightboxImage();
  $("#lbDate").textContent = formatDate(current.created) + ` · ${current.width}×${current.height}`;
  $("#lbChips").innerHTML = chipsHtml(current.counts);
  $("#lbDesc").textContent = current.description;
  $("#lbNote").value = current.note || "";
  $("#lightbox").hidden = false;
  document.body.style.overflow = "hidden";
}

function updateLightboxImage() {
  const file = showRaw ? current.raw_image : current.image;
  $("#lbImg").src = `/media/${file}`;
  $("#lbDownload").href = `/media/${file}`;
  $("#lbDownload").setAttribute("download", file);
  $("#lbToggle").textContent = showRaw ? "Показать с разметкой" : "Показать оригинал";
}

function closeLightbox() {
  $("#lightbox").hidden = true;
  document.body.style.overflow = "";
  current = null;
}

$("#grid").addEventListener("click", (e) => {
  const card = e.target.closest(".shot");
  if (card) openItem(card.dataset.id);
});
$("#gallerySearch").addEventListener("input", render);
$("#lbClose").addEventListener("click", closeLightbox);
$("#lightbox").addEventListener("click", (e) => e.target.id === "lightbox" && closeLightbox());
document.addEventListener("keydown", (e) => {
  if ($("#lightbox").hidden) return;
  if (e.key === "Escape") closeLightbox();
  if (e.key === "ArrowRight" || e.key === "ArrowLeft") {
    const idx = items.indexOf(current) + (e.key === "ArrowRight" ? 1 : -1);
    if (items[idx]) openItem(items[idx].id);
  }
});
$("#lbToggle").addEventListener("click", () => { showRaw = !showRaw; updateLightboxImage(); });

$("#lbSave").addEventListener("click", async () => {
  try {
    const meta = await api(`/api/gallery/${current.id}`, { method: "PATCH", body: { note: $("#lbNote").value } });
    Object.assign(current, meta);
    render();
    toast("Подпись сохранена", "success");
  } catch (e) { toast(escapeHtml(e.message), "error"); }
});

$("#lbDelete").addEventListener("click", async () => {
  if (!confirm("Удалить этот снимок?")) return;
  try {
    await api(`/api/gallery/${current.id}`, { method: "DELETE" });
    items = items.filter((i) => i !== current);
    closeLightbox();
    setGalleryCount(items.length);
    render();
    toast("Снимок удалён");
  } catch (e) { toast(escapeHtml(e.message), "error"); }
});

loadGallery().catch((e) => toast(escapeHtml(e.message), "error"));
