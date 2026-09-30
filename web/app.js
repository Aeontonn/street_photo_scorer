const MAX_BYTES = 10 * 1024 * 1024;
const TYPES = ["image/jpeg", "image/png", "image/webp"];
const CIRCUMFERENCE = 326.7; // 2 * pi * r (r = 52)

const $ = (id) => document.getElementById(id);
const drop = $("drop"), fileInput = $("file"), preview = $("preview");

drop.addEventListener("click", () => fileInput.click());
drop.addEventListener("keydown", (e) => {
  if (e.key === "Enter" || e.key === " ") { e.preventDefault(); fileInput.click(); }
});
drop.addEventListener("dragover", (e) => { e.preventDefault(); drop.classList.add("over"); });
drop.addEventListener("dragleave", () => drop.classList.remove("over"));
drop.addEventListener("drop", (e) => {
  e.preventDefault();
  drop.classList.remove("over");
  if (e.dataTransfer.files[0]) handleFile(e.dataTransfer.files[0]);
});
fileInput.addEventListener("change", () => { if (fileInput.files[0]) handleFile(fileInput.files[0]); });
$("again").addEventListener("click", () => {
  window.scrollTo({ top: 0, behavior: "smooth" });
  fileInput.click();
});

function showError(msg) {
  $("error").textContent = msg;
  $("error").hidden = !msg;
}

async function handleFile(file) {
  showError("");
  if (!TYPES.includes(file.type)) return showError("Please choose a JPG, PNG or WebP image.");
  if (file.size > MAX_BYTES) return showError("That image is over 10 MB. Please choose a smaller one.");

  preview.src = URL.createObjectURL(file);
  preview.hidden = false;
  $("dropHint").hidden = true;
  $("result").hidden = true;
  $("loading").hidden = false;

  try {
    const body = new FormData();
    body.append("file", file);
    const res = await fetch(`${window.API_URL}/score`, { method: "POST", body });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(data.detail || "The server had a problem. Please try again.");
    render(data);
  } catch (err) {
    showError(err instanceof TypeError
      ? "Could not reach the scoring server. If it was asleep, wait a minute and try again."
      : err.message);
  } finally {
    $("loading").hidden = true;
  }
}

function render(d) {
  $("demoBadge").hidden = !d.demo;
  $("score").textContent = d.aesthetic_score.toFixed(1);
  $("label").textContent = d.score_label;
  $("summary").textContent =
    `Better than ${Math.round(d.percentile)}% of the training photos. Closest style: ${d.cluster_name}. ` +
    `The classifier rates it ${d.quality_verdict} quality (${Math.round(d.confidence * 100)}% confident).`;

  $("tags").replaceChildren(...d.attributes.map((a) => {
    const el = document.createElement("span");
    el.className = "tag";
    el.textContent = a.label;
    return el;
  }));

  $("genres").replaceChildren(...d.genre_scores.slice(0, 4).map((g) => {
    const el = document.createElement("div");
    el.className = "genre";
    const name = document.createElement("span");
    const pct = document.createElement("span");
    name.textContent = g.label;
    pct.textContent = `${Math.round(g.score * 100)}%`;
    const bar = document.createElement("div");
    const fill = document.createElement("i");
    bar.className = "bar";
    fill.style.width = `${Math.round(g.score * 100)}%`;
    bar.append(fill);
    el.append(name, pct, bar);
    return el;
  }));

  $("result").hidden = false;
  const ring = $("ringFg");
  ring.style.strokeDashoffset = CIRCUMFERENCE;
  requestAnimationFrame(() => requestAnimationFrame(() => {
    ring.style.strokeDashoffset = CIRCUMFERENCE * (1 - d.aesthetic_score / 10);
  }));
  $("result").scrollIntoView({ behavior: "smooth", block: "start" });
}
