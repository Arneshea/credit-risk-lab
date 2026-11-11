const $ = (selector) => document.querySelector(selector);
const statusEl = $("#benchmark-status");
let benchmarkRequest = 0;

function setStatus(message, isError = false) {
  statusEl.textContent = message;
  statusEl.classList.toggle("error", isError);
}

async function postJson(path, body) {
  const response = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || "The request could not be completed.");
  return data;
}

document.querySelectorAll(".mode-button").forEach((button) => {
  button.addEventListener("click", () => {
    const legacy = button.dataset.mode === "legacy";
    document.querySelectorAll(".mode-button").forEach((item) => item.classList.toggle("active", item === button));
    $("#lab-view").classList.toggle("active-view", !legacy);
    $("#legacy-view").classList.toggle("active-view", legacy);
    $("#legacy-view").hidden = !legacy;
    $("#lab-view").hidden = legacy;
    window.scrollTo({ top: 0, behavior: "smooth" });
  });
});

function percent(value) { return `${(value * 100).toFixed(1)}%`; }
function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, (char) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  })[char]);
}

function renderComparison(data) {
  const [fuzzy, ann] = data.models;
  $("#dataset-summary").innerHTML = [
    `${data.dataset_name}`, `${data.n_train.toLocaleString()} train`,
    `${data.n_validation.toLocaleString()} validation`, `${data.n_test.toLocaleString()} held-out test`,
    `Default rate ${percent(data.default_rate)}`,
  ].map((item) => `<span class="summary-chip">${item}</span>`).join("");
  $("#dataset-summary").classList.remove("hidden");
  $("#method-summary").innerHTML = `<span class="notice-icon">↗</span><p><strong>Hybrid training:</strong> ${data.method.neuro_fuzzy} Each model’s decision threshold is chosen on validation data; the metrics shown are from held-out test data.</p>`;
  $("#method-summary").classList.remove("hidden");

  const aucGap = Math.abs(fuzzy.metrics.auc - ann.metrics.auc);
  const compactness = `${fuzzy.rule_count} rules · ${fuzzy.avg_conditions.toFixed(1)} conditions / rule`;
  const metricCards = [
    ["GA-ANFIS AUC", percent(fuzzy.metrics.auc), "Held-out test ranking"],
    ["ANN AUC", percent(ann.metrics.auc), "Held-out test ranking"],
    ["AUC GAP", `${(aucGap * 100).toFixed(1)} pts`, "Absolute gap between models"],
    ["NEURO-FUZZY RULES", `${fuzzy.rule_count} rules`, `${fuzzy.avg_conditions.toFixed(1)} average conditions per rule`],
  ];
  $("#metric-grid").innerHTML = metricCards.map(([label, value, note]) =>
    `<article class="metric-card"><div class="metric-label">${label}</div><div class="metric-value">${value}</div><div class="metric-note">${note}</div></article>`).join("");

  const scoreMetrics = [
    ["AUC", "auc"], ["Balanced accuracy", "balanced_accuracy"], ["Default recall", "recall"],
  ];
  const makeCard = (model, annModel = false) => {
    const m = model.metrics;
    const badge = annModel ? "badge-ann" : "badge-fuzzy";
    const metrics = scoreMetrics.map(([name, key]) =>
      `<div class="score-item"><span>${name}</span><strong>${percent(m[key])}</strong><div class="score-track"><i class="score-fill" style="width:${Math.max(0, Math.min(100, m[key] * 100))}%"></i></div></div>`).join("");
    const extras = annModel
      ? `<div class="model-foot"><span>Training stopped after ${model.epochs} epochs</span><span>2 hidden layers</span></div>`
      : `<div class="model-foot"><span>${compactness}</span><span>GA ${model.evolution_generations} gens · BP ${model.backprop_epochs} epochs</span></div><div class="rules">${model.rules.map((rule) => `<div class="rule-row"><b>IF</b> ${rule.text.replace(/^IF /, "")} <small>(learned rule output ${percent(rule.risk_contribution)})</small></div>`).join("")}</div>`;
    return `<article class="model-card ${annModel ? "ann-card" : ""}"><header class="model-card-header"><div><h3 class="model-title">${model.name}</h3><div class="model-family">${model.family}</div></div><span class="model-badge ${badge}">${annModel ? "MLP BASELINE" : "RULES + LEARNED SETS"}</span></header><div class="model-card-body"><p class="model-detail">${model.details}</p><div class="model-score-row">${metrics}</div>${extras}</div></article>`;
  };
  $("#model-grid").innerHTML = makeCard(fuzzy) + makeCard(ann, true);
}

async function runBenchmark() {
  const requestId = ++benchmarkRequest;
  const dataset = $("#dataset-select").value;
  const button = $("#run-benchmark");
  button.disabled = true;
  button.innerHTML = `Training models… <span>◌</span>`;
  setStatus("Searching fuzzy rules, fine-tuning with backpropagation, and fitting the ANN baseline…");
  try {
    const data = await postJson("/api/benchmark", { dataset });
    if (requestId !== benchmarkRequest) return;
    renderComparison(data);
    setStatus(`${data.method.split}. Metrics below use validation-selected thresholds.`);
  } catch (error) {
    if (requestId === benchmarkRequest) setStatus(error.message, true);
  } finally {
    if (requestId === benchmarkRequest) {
      button.disabled = false;
      button.innerHTML = `Run model comparison <span>→</span>`;
    }
  }
}

$("#run-benchmark").addEventListener("click", runBenchmark);
$("#dataset-select").addEventListener("change", runBenchmark);

const statusOptions = [
  [-2, "No usage that month"], [-1, "Paid in full"], [0, "Paid minimum only (revolving)"],
  ...Array.from({ length: 8 }, (_, index) => [index + 1, `${index + 1} month${index ? "s" : ""} late`]),
];
$("#month-rows").innerHTML = Array.from({ length: 6 }, (_, index) => {
  const month = index + 1;
  const options = statusOptions.map(([value, label]) => `<option value="${value}" ${value === 0 ? "selected" : ""}>${label}</option>`).join("");
  return `<tr><td>MONTH ${month}</td><td><select name="PAY_${month}" aria-label="Month ${month} payment status">${options}</select></td><td><input name="BILL_AMT${month}" type="number" min="0" max="10000000" step="500" value="20000" aria-label="Month ${month} bill amount" required></td><td><input name="PAY_AMT${month}" type="number" min="0" max="10000000" step="500" value="2000" aria-label="Month ${month} amount paid" required></td></tr>`;
}).join("");

$("#legacy-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const form = event.currentTarget;
  const formData = new FormData(form);
  const payload = {
    customer_ref: $("#customer-ref").value,
    requested_amount: Number($("#requested-amount").value),
    LIMIT_BAL: Number($("#credit-limit").value),
  };
  for (const [key, value] of formData.entries()) payload[key] = Number(value);
  const button = form.querySelector("button[type=submit]");
  const legacyStatus = $("#legacy-status");
  const result = $("#legacy-result");
  legacyStatus.classList.remove("error");
  button.disabled = true;
  button.textContent = "Scoring…";
  legacyStatus.textContent = "Applying the saved fuzzy rule model…";
  result.classList.add("hidden");
  try {
    const data = await postJson("/api/legacy-predict", payload);
    const lowRisk = data.decision === "Approve";
    const reasons = data.reasons.length
      ? data.reasons.map((item) => `<div class="explanation-line">${item.supports_default ? "▲" : "✓"} ${item.text}<small>${item.rule} · firing strength ${item.strength.toFixed(2)}</small></div>`).join("")
      : `<div class="explanation-line">No strong rule fired, so the score stays close to the training-set average.</div>`;
    const tips = data.tips.length
      ? data.tips.map((item) => `<div class="explanation-line">↗ ${item.text}</div>`).join("")
      : `<div class="explanation-line">No model-generated suggestions for this profile.</div>`;
    const safeRef = escapeHtml(data.customer_ref);
    result.innerHTML = `<div class="result-title-row"><div><div class="eyebrow">SAVED TAIWAN FUZZY-GA MODEL</div><h2>${safeRef ? `Application: ${safeRef}` : "Model result"}</h2></div><span class="risk-pill ${lowRisk ? "risk-low" : "risk-high"}">${lowRisk ? "LOWER RISK INDICATION" : "REFER FOR REVIEW"}</span></div><div class="result-cards"><div class="result-card"><span>Requested amount · display only</span><strong>₹${data.requested_amount.toLocaleString("en-IN")}</strong></div><div class="result-card"><span>Default risk estimate</span><strong>${percent(data.risk)}</strong></div><div class="result-card"><span>Credit score</span><strong>${data.credit_score} / 100</strong></div></div><div class="result-explain"><h3>Why the model gave this result</h3>${reasons}<h3>What-if suggestions from the model</h3>${tips}</div><p class="result-note">This is a research demonstration, not a loan approval. The requested amount is not used in the risk calculation.</p>`;
    result.classList.remove("hidden");
    legacyStatus.textContent = "Prediction updated from the submitted form values.";
    result.scrollIntoView({ behavior: "smooth", block: "start" });
  } catch (error) {
    legacyStatus.textContent = error.message;
    legacyStatus.classList.add("error");
  } finally {
    button.disabled = false;
    button.innerHTML = `Predict risk <span>→</span>`;
  }
});

runBenchmark();
