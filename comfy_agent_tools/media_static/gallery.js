const state = {
  items: [],
  selected: null,
};

const itemsEl = document.getElementById("items");
const stageEl = document.getElementById("stage");
const infoEl = document.getElementById("info");
const searchEl = document.getElementById("search");
const kindEl = document.getElementById("kind");

async function load() {
  const response = await fetch("/api/index");
  const index = await response.json();
  state.items = index.items || [];
  state.selected = state.items[0] || null;
  render();
}

function filteredItems() {
  const search = searchEl.value.trim().toLowerCase();
  const kind = kindEl.value;
  return state.items.filter((item) => {
    if (kind && item.kind !== kind) return false;
    if (!search) return true;
    const haystack = [item.title, item.prompt, item.profile, item.mode, item.tool].filter(Boolean).join(" ").toLowerCase();
    return haystack.includes(search);
  });
}

function render() {
  const items = filteredItems();
  if (!items.includes(state.selected)) {
    state.selected = items[0] || null;
  }
  itemsEl.innerHTML = items.map((item) => card(item)).join("");
  for (const button of itemsEl.querySelectorAll("button[data-id]")) {
    button.addEventListener("click", () => {
      state.selected = state.items.find((item) => item.id === button.dataset.id);
      render();
      button.scrollIntoView({ behavior: "smooth", block: "nearest", inline: "center" });
    });
  }
  stageEl.innerHTML = stage(state.selected);
  infoEl.innerHTML = info(state.selected);
}

function card(item) {
  const selected = state.selected?.id === item.id ? " selected" : "";
  return `<button class="item${selected}" data-id="${escapeAttr(item.id)}">
    <div class="thumb">${media(item)}</div>
    <div class="body">
      <div class="title">${escapeHtml(item.title || "media")}</div>
      <div class="meta">${escapeHtml([item.kind, item.mode, item.profile].filter(Boolean).join(" · "))}</div>
    </div>
  </button>`;
}

function stage(item) {
  if (!item) return `<div class="empty">No media found.</div>`;
  return `<div class="stage-preview">${media(item, true)}</div>`;
}

function info(item) {
  if (!item) return "<p>No media found.</p>";
  const artifact = item.artifacts?.[0]?.path || "";
  return `<h2>${escapeHtml(item.title || "media")}</h2>
    <dl>
      <dt>Kind</dt><dd>${escapeHtml(item.kind || "")}</dd>
      <dt>Mode</dt><dd>${escapeHtml(item.mode || "")}</dd>
      <dt>Profile</dt><dd>${escapeHtml(item.profile || "")}</dd>
      <dt>Seed</dt><dd>${escapeHtml(String(item.seed ?? ""))}</dd>
      <dt>Prompt</dt><dd>${escapeHtml(item.prompt || "")}</dd>
      <dt>Artifact</dt><dd>${escapeHtml(artifact)}</dd>
    </dl>`;
}

function media(item, controls = false) {
  const url = item.media_url;
  if (!url) return `<span>${escapeHtml(item.kind || "media")}</span>`;
  if (item.kind === "image") return `<img src="${escapeAttr(url)}" alt="">`;
  if (item.kind === "video") {
    const width = Number(item.hyperframes?.width || 0);
    const height = Number(item.hyperframes?.height || 0);
    const style = controls && width > 0 && height > 0 ? ` style="--media-aspect-ratio: ${width} / ${height}"` : "";
    return `<video src="${escapeAttr(url)}"${style} ${controls ? "controls autoplay loop" : ""} muted playsinline></video>`;
  }
  if (item.kind === "music") return `<audio src="${escapeAttr(url)}" controls></audio>`;
  if (item.kind === "hyperframes") return `<hyperframes-player src="${escapeAttr(url)}" controls></hyperframes-player>`;
  return `<span>${escapeHtml(item.kind || "media")}</span>`;
}

function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, (char) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#039;" })[char]);
}

function escapeAttr(value) {
  return escapeHtml(value);
}

searchEl.addEventListener("input", render);
kindEl.addEventListener("change", render);
load();
