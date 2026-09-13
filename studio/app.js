/* nanoforge studio — vanilla JS, no frameworks, no CDN (offline is the brand) */
"use strict";

const $ = (id) => document.getElementById(id);
let state = { projects: [], templates: [], project: null, current: null, lastAgentMsg: "" };

async function api(path, opts = {}) {
  const res = await fetch(`/api/${path}`, {
    headers: { "Content-Type": "application/json" },
    ...opts,
    body: opts.body ? JSON.stringify(opts.body) : undefined,
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.error || res.statusText);
  return data;
}

/* ---------- boot ---------- */
window.addEventListener("DOMContentLoaded", boot);

async function boot() {
  document.querySelectorAll(".tab").forEach((t) =>
    t.addEventListener("click", () => switchTab(t.dataset.tab))
  );
  $("btnSaveSpec").addEventListener("click", saveSpec);
  $("btnTrain").addEventListener("click", startTrain);
  $("btnStop").addEventListener("click", () => api("stop", { method: "POST", body: {} }));
  $("btnExport").addEventListener("click", exportRun);
  $("btnChat").addEventListener("click", sendChat);
  $("chatBox").addEventListener("keydown", (e) => e.key === "Enter" && sendChat());
  $("btnNewProject").addEventListener("click", openNewProject);
  $("btnCreateProject").addEventListener("click", createProject);
  $("btnApplyDiff").addEventListener("click", applyDiff);
  $("projectSelect").addEventListener("change", (e) => loadProject(e.target.value));
  document.querySelectorAll("[data-add]").forEach((b) =>
    b.addEventListener("click", () => specEdit({ op: "add_block", kind: b.dataset.add }))
  );

  const st = await api("state");
  state.projects = st.projects;
  state.templates = st.templates;
  $("ver").textContent = "v" + st.version;
  fillSelect($("projectSelect"), st.projects, st.projects[0] || "");
  fillSelect($("npTemplate"), st.templates, "tiny-char-lm");
  if (st.projects.length) await loadProject(st.projects[0]);
  setInterval(pollActive, 900);
  await pollActive();
}

function fillSelect(sel, items, value) {
  sel.innerHTML = "";
  for (const it of items) {
    const o = document.createElement("option");
    o.value = o.textContent = it;
    sel.appendChild(o);
  }
  if (value) sel.value = value;
}

function switchTab(name) {
  document.querySelectorAll(".tab").forEach((t) => t.classList.toggle("active", t.dataset.tab === name));
  document.querySelectorAll(".tab-page").forEach((p) => p.classList.toggle("active", p.id === `tab-${name}`));
}

/* ---------- project ---------- */
async function loadProject(name) {
  state.project = name;
  $("projectSelect").value = name;
  state.current = await api(`project?name=${encodeURIComponent(name)}`);
  renderProject();
}

function renderProject() {
  const c = state.current;
  $("specEditor").value = c.spec_yaml;
  setValidation($("specValidation"), c.validation);
  $("sideStack").textContent = c.spec_struct ? c.spec_struct.stack || c.stack : "";
  $("sideStack").textContent = c.stack || "—";
  $("sideParams").textContent =
    c.params ? `${(c.params / 1e6).toFixed(2)}M params` : "";
  const eb = $("sideExport");
  if (c.exportable) {
    eb.textContent = "✓ exportable to nanollama.c";
    eb.className = "export-badge yes";
  } else {
    eb.textContent = "▲ studio-only: " + (c.export_block || "not exportable");
    eb.className = "export-badge no";
  }
  const fl = $("fileList");
  fl.innerHTML = "";
  for (const f of c.files) {
    const li = document.createElement("li");
    li.textContent = f;
    li.addEventListener("click", () => previewFile(f, li));
    fl.appendChild(li);
  }
  renderRunSidebar();
  renderCanvas();
  fillShipRuns();
}

async function previewFile(path, li) {
  document.querySelectorAll("#fileList li").forEach((x) => x.classList.remove("active"));
  li.classList.add("active");
  const f = await api(`file?project=${encodeURIComponent(state.project)}&path=${encodeURIComponent(path)}`);
  switchTab("spec");
  $("specEditor").value = f.binary
    ? `# ${f.path}\n# binary artifact (${(f.size / 1e6).toFixed(2)} MB)`
    : f.content;
  setValidation($("specValidation"), { ok: true, error: `read-only preview: ${path}` });
}

function setValidation(el, v) {
  el.textContent = v.ok ? "✓ valid nanospec" : "✗ " + v.error;
  el.className = "validation " + (v.ok ? "ok" : "bad");
}

/* ---------- spec editor ---------- */
async function saveSpec() {
  const r = await api("project", {
    method: "PUT",
    body: { project: state.project, spec_yaml: $("specEditor").value },
  });
  setValidation($("specValidation"), r.validation);
  await loadProject(state.project);
}

/* ---------- canvas ---------- */
async function specEdit(body) {
  await api("spec_edit", { method: "POST", body: { project: state.project, ...body } });
  const c = await api(`project?name=${encodeURIComponent(state.project)}`);
  state.current = c;
  $("specEditor").value = c.spec_yaml;
  setValidation($("canvasValidation"), c.validation);
  renderProject();
}

function renderCanvas() {
  const host = $("canvasStack");
  host.innerHTML = "";
  const s = state.current.spec_struct;
  if (!s) return;
  const trainBox = document.createElement("div");
  trainBox.className = "block-card";
  trainBox.innerHTML = `<span class="kind other">train</span>
    <label>steps <input data-train="steps" type="number" value="${s.train.steps}"></label>
    <label>batch <input data-train="batch" type="number" value="${s.train.batch}"></label>
    <label>lr <input data-train="lr" type="number" step="0.0001" value="${s.train.lr}"></label>
    <label>warmup <input data-train="warmup" type="number" value="${s.train.warmup}"></label>`;
  host.appendChild(trainBox);
  trainBox.querySelectorAll("input").forEach((inp) =>
    inp.addEventListener("change", () =>
      specEdit({ op: "set_train", key: inp.dataset.train, value: inp.value })
    )
  );

  s.layers.forEach((b, i) => {
    const card = document.createElement("div");
    card.className = "block-card";
    const cls = ["attention", "swiglu_ffn"].includes(b.kind) ? b.kind : "other";
    let params = "";
    if (b.kind === "attention")
      params = `<label>heads <input data-i="${i}" data-key="heads" type="number" value="${b.params.heads ?? s.heads}"></label>`;
    if (["swiglu_ffn", "gelu_ffn"].includes(b.kind))
      params = `<label>hidden <input data-i="${i}" data-key="hidden" type="number" value="${b.params.hidden ?? s.hidden}"></label>`;
    if (b.kind === "conv1d")
      params = `<label>kernel <input data-i="${i}" data-key="kernel" type="number" value="${b.params.kernel ?? 3}"></label>`;
    card.innerHTML = `
      <span class="idx">${i}</span>
      <span class="kind ${cls}">${b.kind}</span>
      ${params}
      <span class="spacer"></span>
      <button class="ghost" data-move="up" data-i="${i}">↑</button>
      <button class="ghost" data-move="down" data-i="${i}">↓</button>
      <button class="danger" data-del="${i}">×</button>`;
    host.appendChild(card);
    if (i < s.layers.length - 1) {
      const arrow = document.createElement("div");
      arrow.style.cssText = "text-align:center;color:var(--dim)";
      arrow.textContent = "↓";
      host.appendChild(arrow);
    }
  });

  host.querySelectorAll("input[data-key]").forEach((inp) =>
    inp.addEventListener("change", () =>
      specEdit({ op: "set_block_param", index: +inp.dataset.i, key: inp.dataset.key, value: inp.value })
    )
  );
  host.querySelectorAll("button[data-del]").forEach((b) =>
    b.addEventListener("click", () => specEdit({ op: "remove_block", index: +b.dataset.del }))
  );
  host.querySelectorAll("button[data-move]").forEach((b) =>
    b.addEventListener("click", () =>
      specEdit({ op: "move_block", index: +b.dataset.i, dir: b.dataset.move })
    )
  );
}

/* ---------- training ---------- */
async function startTrain() {
  const body = {
    project: state.project,
    device: $("tDevice").value || null,
    steps: +$("tSteps").value || null,
    batch: +$("tBatch").value || null,
    lr: +$("tLr").value || null,
  };
  try {
    await api("train", { method: "POST", body });
    $("trainStatus").textContent = "starting…";
  } catch (e) {
    $("trainStatus").textContent = e.message;
  }
  pollActive();
}

let wasRunning = false;

async function pollActive() {
  let a;
  try { a = await api("active"); } catch { return; }
  $("btnStop").disabled = !a.running;
  const chart = $("lossChart");
  if (a.running && a.run_dir) {
    wasRunning = true;
    $("trainStatus").textContent = `training ${a.run}…`;
    if (a.metrics && a.metrics.length) drawChart(chart, a.metrics, a.run);
  } else {
    if (wasRunning) {
      $("trainStatus").textContent = "idle — run finished";
      await loadProject(state.project);
    }
    wasRunning = false;
  }
}

function drawChart(canvas, metrics, label) {
  const ctx = canvas.getContext("2d");
  const W = canvas.width, H = canvas.height, pad = 34;
  ctx.clearRect(0, 0, W, H);
  const losses = metrics.filter((m) => m.loss !== undefined);
  const vals = metrics.filter((m) => m.val_loss !== undefined);
  if (!losses.length) return;
  const maxX = losses[losses.length - 1].step;
  const maxL = Math.max(...losses.map((m) => m.loss)) * 1.05;
  const minL = Math.min(...losses.map((m) => m.loss), ...(vals.map((v) => v.val_loss) || [maxL])) * 0.95;
  const X = (s) => pad + (s / Math.max(1, maxX)) * (W - pad - 12);
  const Y = (l) => H - pad - ((l - minL) / Math.max(1e-6, maxL - minL)) * (H - 2 * pad);

  ctx.strokeStyle = "#2a3341"; ctx.fillStyle = "#8b98a9"; ctx.font = "11px monospace";
  for (let i = 0; i <= 4; i++) {
    const l = minL + ((maxL - minL) * i) / 4;
    ctx.beginPath(); ctx.moveTo(pad, Y(l)); ctx.lineTo(W - 12, Y(l)); ctx.stroke();
    ctx.fillText(l.toFixed(2), 2, Y(l) + 4);
  }
  ctx.beginPath(); ctx.strokeStyle = "#f0883e"; ctx.lineWidth = 1.5;
  losses.forEach((m, i) => (i ? ctx.lineTo(X(m.step), Y(m.loss)) : ctx.moveTo(X(m.step), Y(m.loss))));
  ctx.stroke();
  if (vals.length) {
    ctx.beginPath(); ctx.strokeStyle = "#58a6ff"; ctx.lineWidth = 1.5;
    vals.forEach((m, i) => (i ? ctx.lineTo(X(m.step), Y(m.val_loss)) : ctx.moveTo(X(m.step), Y(m.val_loss))));
    ctx.stroke();
  }
  $("trainLegend").innerHTML =
    `<span style="color:var(--accent)">— train loss</span> &nbsp;` +
    (vals.length ? `<span style="color:var(--accent2)">— val loss</span> &nbsp;` : "") +
    `<span>${label} · step ${maxX}</span>`;
}

/* ---------- runs ---------- */
function renderRunSidebar() {
  const ul = $("runList");
  ul.innerHTML = "";
  for (const r of state.current.runs.slice(0, 14)) {
    const li = document.createElement("li");
    const mark = r.status === "done" ? "<b>●</b>" : r.status === "failed" ? "<i>✗</i>" : "◐";
    li.innerHTML = `${mark} ${r.id}`;
    li.title = `${r.status} — click for detail`;
    li.addEventListener("click", () => showRun(r.id));
    ul.appendChild(li);
  }
}

async function showRun(id) {
  switchTab("runs");
  const r = await api(`run?project=${encodeURIComponent(state.project)}&id=${encodeURIComponent(id)}`);
  const el = $("runDetail");
  const finalMetric = r.metrics_log.filter((m) => m.val_loss !== undefined).slice(-1)[0]
    || r.metrics_log.filter((m) => m.val_accuracy !== undefined).slice(-1)[0] || {};
  el.innerHTML = `
    <h4>${r.id} — <span style="color:${r.status === "done" ? "var(--ok)" : "var(--bad)"}">${r.status}</span></h4>
    <pre>${escapeHtml(JSON.stringify({ metrics: r.metrics, error: r.error, note: r.config_note }, null, 2))}</pre>
    ${r.error ? `<h4>error</h4><pre style="color:var(--bad)">${escapeHtml(r.error)}</pre>` : ""}
    ${r.samples ? `<h4>samples while training</h4><pre>${escapeHtml(r.samples)}</pre>` : ""}`;
  const c = $("lossChart");
  drawChart(c, r.metrics_log || [], id);
  switchTab("runs");
}

function fillShipRuns() {
  const done = state.current.runs.filter((r) => r.status === "done");
  fillSelect($("shipRun"), done.map((r) => r.id), done[0]?.id);
}

async function exportRun() {
  const id = $("shipRun").value;
  if (!id) return ($("shipStatus").textContent = "no finished run to export");
  $("shipStatus").textContent = "exporting…";
  try {
    const r = await api("export", { method: "POST", body: { project: state.project, run_id: id } });
    $("exportOut").textContent = r.files.map((f) => "shipped: " + f).join("\n") +
      `\n\nrun it:  nanollama run -m ${r.files[0]} -i "Once upon a time"  (add -q for int8)`;
    const rep = await api(`report?project=${encodeURIComponent(state.project)}&id=${encodeURIComponent(id)}`);
    $("qualityReport").textContent = rep.report;
    $("qualityReport").style.display = "block";
    $("shipStatus").textContent = "";
  } catch (e) {
    $("shipStatus").textContent = e.message;
  }
}

/* ---------- agent ---------- */
function addMsg(role, text, meta) {
  const div = document.createElement("div");
  div.className = `msg ${role}`;
  div.textContent = text;
  if (meta) {
    const m = document.createElement("div");
    m.className = "meta";
    m.textContent = meta;
    div.appendChild(m);
  }
  $("chatLog").appendChild(div);
  $("chatLog").scrollTop = 1e9;
  return div;
}

async function sendChat(confirm = false) {
  const msg = $("chatBox").value.trim() || state.lastAgentMsg;
  if (!msg) return;
  if (!confirm) {
    addMsg("user", msg);
    $("chatBox").value = "";
  }
  state.lastAgentMsg = msg;
  const typing = addMsg("bot", "…");
  try {
    const r = await api("agent", { method: "POST", body: { project: state.project, message: msg, confirm } });
    typing.remove();
    addMsg("bot", r.reply, `backend: ${r.backend}${r.wrote?.length ? " · wrote " + r.wrote.join(", ") : ""}`);
    const preview = r.results?.find((x) => x.kind === "write_spec" && x.status === "preview");
    if (preview) {
      $("diffView").textContent = preview.diff || "(no changes)";
      $("diffModal").classList.remove("hidden");
    }
    if (r.wrote?.length) await loadProject(state.project);
  } catch (e) {
    typing.textContent = "agent error: " + e.message;
  }
}

async function applyDiff() {
  closeModal();
  await sendChat(true);
}

/* ---------- new project ---------- */
function openNewProject() {
  $("npError").textContent = "";
  $("newProjectModal").classList.remove("hidden");
  $("npName").focus();
}
function closeModal() {
  document.querySelectorAll(".modal").forEach((m) => m.classList.add("hidden"));
}
async function createProject() {
  try {
    await api("init_project", { method: "POST", body: { name: $("npName").value.trim(), template: $("npTemplate").value } });
    closeModal();
    const st = await api("state");
    fillSelect($("projectSelect"), st.projects, $("npName").value.trim());
    await loadProject($("npName").value.trim());
  } catch (e) {
    $("npError").textContent = e.message;
  }
}
window.closeModal = closeModal;

function escapeHtml(s) {
  return String(s ?? "").replace(/[&<>]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]));
}
