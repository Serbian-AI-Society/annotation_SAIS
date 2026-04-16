"""
Generate a comprehensive self-contained HTML annotation report for NanoBEIR-sr.

Opens in any browser — no server, no internet connection, no external dependencies.
All data is embedded in the HTML file as JSON.

Features:
  - Summary cards (total, done %, annotators, avg score)
  - Benchmark progress table with inline progress bars
  - Per-annotator panel with score distribution charts
  - Full annotations table: sortable, filterable, searchable
  - Expandable rows showing full English source, Serbian translation,
    correction, and comment side by side

Usage:
    python generate_report.py
    python generate_report.py --output report.html --no-open

Environment variables (or pass as args):
    ARGILLA_API_URL
    ARGILLA_API_KEY
"""

import argparse
import json
import os
import sys
import warnings
import webbrowser
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import argilla as rg

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = open(sys.stdout.fileno(), mode="w", encoding="utf-8", buffering=1)

BENCHMARK_NAMES = [
    "NanoArguAna", "NanoTouche2020", "NanoSciFact", "NanoSCIDOCS",
    "NanoNQ", "NanoNFCorpus", "NanoMSMARCO", "NanoFiQA2018",
    "NanoHotpotQA", "NanoFEVER", "NanoDBPedia", "NanoQuoraRetrieval",
    "NanoClimateFEVER",
]


# ---------------------------------------------------------------------------
# Data fetching
# ---------------------------------------------------------------------------
def fetch_data(client: rg.Argilla, dataset: rg.Dataset) -> dict:
    print("  [1/3] Loading user list...", flush=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        user_map = {str(u.id): u.username for u in client.users}

    print("  [2/3] Counting records by benchmark and status...", flush=True)
    benchmark_stats = {
        name: {"queries_total": 0, "queries_done": 0,
               "passages_total": 0, "passages_done": 0}
        for name in BENCHMARK_NAMES
    }

    for status_val in ("pending", "completed", "discarded"):
        done = status_val in ("completed", "discarded")
        query = rg.Query(filter=rg.Filter([("status", "==", status_val)]))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for rec in dataset.records(query=query, with_responses=False):
                meta = rec.metadata or {}
                bm = meta.get("benchmark", "Unknown")
                rt = meta.get("record_type", "unknown")
                if bm not in benchmark_stats:
                    continue
                if rt == "query":
                    benchmark_stats[bm]["queries_total"] += 1
                    if done:
                        benchmark_stats[bm]["queries_done"] += 1
                elif rt == "passage":
                    benchmark_stats[bm]["passages_total"] += 1
                    if done:
                        benchmark_stats[bm]["passages_done"] += 1

    print("  [3/3] Loading annotations with full text...", flush=True)
    annotations = []
    annotator_stats = defaultdict(lambda: {
        "total": 0, "scores": defaultdict(int), "discarded": 0
    })

    for status_val in ("completed", "discarded"):
        query = rg.Query(filter=rg.Filter([("status", "==", status_val)]))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for rec in dataset.records(query=query, with_responses=True):
                meta = rec.metadata or {}
                fields = rec._model.fields or {}
                source_en = fields.get("source_text_en") or ""
                translation_sr = fields.get("translated_text_sr") or ""
                benchmark = meta.get("benchmark", "Unknown")
                record_type = meta.get("record_type", "unknown")
                task_id = meta.get("task_id", "")
                rec_id = rec._model.external_id or str(rec._model.id)
                updated_at = rec._model.updated_at
                date_str = updated_at.strftime("%Y-%m-%d %H:%M") if updated_at else ""

                for resp in (rec._model.responses or []):
                    uid = str(resp.user_id)
                    username = user_map.get(uid, uid[:8])

                    if resp.status.value == "discarded":
                        annotator_stats[username]["discarded"] += 1
                        annotator_stats[username]["total"] += 1
                        annotations.append({
                            "id": rec_id,
                            "annotator": username,
                            "benchmark": benchmark,
                            "record_type": record_type,
                            "task_id": task_id,
                            "score": "-",
                            "score_label": "Discarded",
                            "source_en": source_en,
                            "translation_sr": translation_sr,
                            "correction": "",
                            "comment": "",
                            "date": date_str,
                        })
                    elif resp.status.value == "submitted":
                        vals = resp.values or {}
                        score_raw = (vals.get("quality_score") or {}).get("value", "")
                        score_num = str(score_raw).strip()[0] if score_raw and str(score_raw).strip() else "?"
                        if not score_num.isdigit():
                            score_num = "?"
                        correction = (vals.get("corrected_text_sr") or {}).get("value", "") or ""
                        comment = (vals.get("comment") or {}).get("value", "") or ""

                        annotator_stats[username]["scores"][score_num] += 1
                        annotator_stats[username]["total"] += 1

                        annotations.append({
                            "id": rec_id,
                            "annotator": username,
                            "benchmark": benchmark,
                            "record_type": record_type,
                            "task_id": task_id,
                            "score": score_num,
                            "score_label": str(score_raw).strip() if score_raw else "?",
                            "source_en": source_en,
                            "translation_sr": translation_sr,
                            "correction": correction,
                            "comment": comment,
                            "date": date_str,
                        })

    # Sort annotations newest first
    annotations.sort(key=lambda x: x["date"], reverse=True)

    # Convert defaultdicts to plain dicts for JSON serialisation
    annotator_stats_clean = {
        u: {"total": v["total"], "scores": dict(v["scores"]), "discarded": v["discarded"]}
        for u, v in sorted(annotator_stats.items(), key=lambda x: -x[1]["total"])
    }

    total_records = sum(
        s["queries_total"] + s["passages_total"] for s in benchmark_stats.values()
    )
    total_done = sum(
        s["queries_done"] + s["passages_done"] for s in benchmark_stats.values()
    )
    submitted = [a for a in annotations if a["score"].isdigit()]
    avg_score = (
        round(sum(int(a["score"]) for a in submitted) / len(submitted), 2)
        if submitted else None
    )

    return {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "dataset": dataset.name,
        "total_records": total_records,
        "total_done": total_done,
        "avg_score": avg_score,
        "benchmark_names": BENCHMARK_NAMES,
        "benchmark_stats": benchmark_stats,
        "annotator_stats": annotator_stats_clean,
        "annotations": annotations,
    }


# ---------------------------------------------------------------------------
# HTML template
# ---------------------------------------------------------------------------
HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="sr">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>NanoBEIR-sr &mdash; Annotation Report</title>
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         background: #f0f2f5; color: #1a1a2e; font-size: 14px; }

  /* Layout */
  .page { max-width: 1400px; margin: 0 auto; padding: 24px 16px; }
  header { background: #1e3a5f; color: white; padding: 20px 24px;
           border-radius: 10px; margin-bottom: 24px;
           display: flex; align-items: center; justify-content: space-between; }
  header h1 { font-size: 22px; font-weight: 700; letter-spacing: -0.3px; }
  header .meta { font-size: 12px; opacity: 0.75; text-align: right; }

  /* Cards */
  .cards { display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin-bottom: 24px; }
  .card { background: white; border-radius: 10px; padding: 20px 24px;
          box-shadow: 0 1px 4px rgba(0,0,0,.08); }
  .card-value { font-size: 32px; font-weight: 700; color: #1e3a5f; line-height: 1; }
  .card-label { font-size: 12px; color: #64748b; margin-top: 6px; text-transform: uppercase;
                letter-spacing: .5px; }
  .card-sub { font-size: 13px; color: #94a3b8; margin-top: 2px; }

  /* Sections */
  section { background: white; border-radius: 10px; padding: 24px;
            box-shadow: 0 1px 4px rgba(0,0,0,.08); margin-bottom: 24px; }
  section h2 { font-size: 16px; font-weight: 600; color: #1e3a5f;
               margin-bottom: 18px; padding-bottom: 10px;
               border-bottom: 2px solid #e2e8f0; }

  /* Tables */
  table { width: 100%; border-collapse: collapse; }
  th { background: #f8fafc; text-align: left; padding: 10px 12px;
       font-size: 11px; text-transform: uppercase; letter-spacing: .5px;
       color: #64748b; border-bottom: 2px solid #e2e8f0; white-space: nowrap; }
  th.sortable { cursor: pointer; user-select: none; }
  th.sortable:hover { color: #1e3a5f; }
  th .sort-icon { opacity: 0.4; font-size: 10px; }
  th.sort-asc .sort-icon, th.sort-desc .sort-icon { opacity: 1; color: #2563eb; }
  td { padding: 10px 12px; border-bottom: 1px solid #f1f5f9; vertical-align: middle; }
  tr:last-child td { border-bottom: none; }
  tr:hover td { background: #f8fafc; }
  tr.expanded td { background: #eff6ff; }

  /* Progress bar */
  .bar-wrap { width: 100px; height: 8px; background: #e2e8f0;
              border-radius: 4px; display: inline-block; vertical-align: middle; }
  .bar-fill { height: 100%; border-radius: 4px; background: #2563eb; transition: width .3s; }
  .bar-fill.complete { background: #16a34a; }
  .pct { font-size: 12px; color: #64748b; margin-left: 8px; }

  /* Score badges */
  .badge { display: inline-block; padding: 2px 10px; border-radius: 12px;
           font-size: 12px; font-weight: 600; white-space: nowrap; }
  .score-1 { background: #fee2e2; color: #991b1b; }
  .score-2 { background: #ffedd5; color: #9a3412; }
  .score-3 { background: #fef9c3; color: #854d0e; }
  .score-4 { background: #dcfce7; color: #166534; }
  .score-5 { background: #bbf7d0; color: #14532d; }
  .score-d { background: #f1f5f9; color: #64748b; }
  .score-q { background: #f1f5f9; color: #64748b; }

  /* Type badge */
  .type-badge { font-size: 11px; padding: 2px 8px; border-radius: 10px;
                font-weight: 600; text-transform: uppercase; }
  .type-query   { background: #dbeafe; color: #1e40af; }
  .type-passage { background: #f3e8ff; color: #6b21a8; }

  /* Score distribution bar (annotator table) */
  .dist-bar { display: flex; height: 14px; border-radius: 4px; overflow: hidden;
              min-width: 80px; width: 120px; }
  .dist-seg { height: 100%; transition: width .3s; }

  /* Expanded row */
  .expand-row td { padding: 0; border-bottom: 2px solid #bfdbfe; }
  .expand-content { padding: 20px 24px; background: #eff6ff;
                    display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
  .expand-col h4 { font-size: 11px; text-transform: uppercase; letter-spacing: .5px;
                   color: #64748b; margin-bottom: 8px; }
  .expand-col .text-box { background: white; border: 1px solid #e2e8f0; border-radius: 8px;
                           padding: 14px; font-size: 13px; line-height: 1.65;
                           white-space: pre-wrap; word-break: break-word;
                           max-height: 260px; overflow-y: auto; color: #1e293b; }
  .text-box.no-correction { color: #94a3b8; font-style: italic; }
  .text-box.correction-made { border-color: #86efac; }

  /* Filters */
  .filters { display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 18px; align-items: center; }
  .filters select, .filters input { border: 1px solid #e2e8f0; border-radius: 8px;
    padding: 7px 12px; font-size: 13px; color: #1e293b; background: #f8fafc;
    outline: none; }
  .filters select:focus, .filters input:focus { border-color: #2563eb;
    box-shadow: 0 0 0 3px rgba(37,99,235,.1); }
  .filters input { min-width: 220px; }
  .count-label { margin-left: auto; font-size: 13px; color: #64748b; }

  /* Truncated text */
  .trunc { max-width: 260px; overflow: hidden; text-overflow: ellipsis;
           white-space: nowrap; display: block; }
  .no-corr { color: #94a3b8; font-style: italic; }
  .click-hint { font-size: 11px; color: #94a3b8; margin-top: 4px; }

  /* Empty state */
  .empty { text-align: center; padding: 40px; color: #94a3b8; font-size: 14px; }
</style>
</head>
<body>
<div class="page">

<header>
  <div>
    <h1>&#128202; NanoBEIR-sr &mdash; Annotation Report</h1>
    <div style="font-size:13px;opacity:.85;margin-top:4px;" id="hdr-dataset"></div>
  </div>
  <div class="meta" id="hdr-meta"></div>
</header>

<!-- Summary cards -->
<div class="cards" id="cards"></div>

<!-- Benchmark progress -->
<section>
  <h2>Benchmark Progress</h2>
  <table id="bm-table">
    <thead><tr>
      <th>Benchmark</th>
      <th>Progress</th>
      <th>Queries</th>
      <th>Passages</th>
      <th>Total done</th>
    </tr></thead>
    <tbody id="bm-body"></tbody>
  </table>
</section>

<!-- Per-annotator -->
<section>
  <h2>Annotators</h2>
  <table id="ann-table">
    <thead><tr>
      <th>Annotator</th>
      <th>Total</th>
      <th style="min-width:140px">Score distribution</th>
      <th>Avg score</th>
      <th>1</th><th>2</th><th>3</th><th>4</th><th>5</th>
      <th>Skipped</th>
    </tr></thead>
    <tbody id="ann-body"></tbody>
  </table>
</section>

<!-- Annotations -->
<section>
  <h2>All Annotations</h2>
  <div class="filters">
    <select id="f-annotator"><option value="">All annotators</option></select>
    <select id="f-benchmark"><option value="">All benchmarks</option></select>
    <select id="f-type"><option value="">All types</option>
      <option value="query">Query</option>
      <option value="passage">Passage</option>
    </select>
    <select id="f-score"><option value="">All scores</option>
      <option value="1">1 &ndash; Potpuno neta&#269;an</option>
      <option value="2">2 &ndash; Ve&#263;e gre&#353;ke</option>
      <option value="3">3 &ndash; Adekvatan</option>
      <option value="4">4 &ndash; Dobar</option>
      <option value="5">5 &ndash; Odli&#269;an</option>
      <option value="-">Discarded</option>
    </select>
    <input id="f-search" type="text" placeholder="&#128269; Search text...">
    <span class="count-label" id="ann-count"></span>
  </div>
  <div class="click-hint" style="margin-bottom:12px">&#9660; Click any row to expand full text</div>
  <table id="ann-tbl">
    <thead><tr>
      <th style="width:36px">#</th>
      <th class="sortable" data-col="annotator">Annotator <span class="sort-icon">&#8597;</span></th>
      <th class="sortable" data-col="benchmark">Benchmark <span class="sort-icon">&#8597;</span></th>
      <th class="sortable" data-col="record_type">Type <span class="sort-icon">&#8597;</span></th>
      <th class="sortable" data-col="score">Score <span class="sort-icon">&#8597;</span></th>
      <th>English source</th>
      <th>Correction</th>
      <th>Comment</th>
      <th class="sortable" data-col="date">Date <span class="sort-icon">&#8597;</span></th>
    </tr></thead>
    <tbody id="ann-body2"></tbody>
  </table>
  <div class="empty" id="ann-empty" style="display:none">No annotations match the current filters.</div>
</section>

</div><!-- .page -->

<script>
const DATA = __DATA_JSON__;

const SCORE_COLOR = {"1":"score-1","2":"score-2","3":"score-3","4":"score-4","5":"score-5","-":"score-d","?":"score-q"};
const SCORE_HEX   = {"1":"#dc2626","2":"#ea580c","3":"#ca8a04","4":"#65a30d","5":"#16a34a"};

function esc(s) {
  return String(s||"").replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;").replace(/"/g,"&quot;");
}
function trunc(s, n) { s=s||""; return s.length>n ? s.slice(0,n)+"…" : s; }

// ---- Header ----
document.getElementById("hdr-dataset").textContent = "Dataset: " + DATA.dataset;
document.getElementById("hdr-meta").innerHTML =
  "Generated: " + DATA.generated_at + "<br>Total records: " + DATA.total_records;

// ---- Cards ----
(function(){
  const pct = DATA.total_records ? Math.round(DATA.total_done*100/DATA.total_records) : 0;
  const annotators = Object.keys(DATA.annotator_stats).length;
  const submitted = DATA.annotations.filter(a => /^[1-5]$/.test(a.score));
  const avgScore = DATA.avg_score ? DATA.avg_score.toFixed(2) : "—";
  const cards = [
    {v: DATA.total_records, l: "Total records", s: ""},
    {v: DATA.total_done + " (" + pct + "%)", l: "Annotated", s: "of " + DATA.total_records},
    {v: annotators, l: "Annotators", s: submitted.length + " submissions"},
    {v: avgScore, l: "Avg quality score", s: "across all submissions"},
  ];
  document.getElementById("cards").innerHTML = cards.map(c =>
    `<div class="card"><div class="card-value">${esc(c.v)}</div>
     <div class="card-label">${esc(c.l)}</div>
     <div class="card-sub">${esc(c.s)}</div></div>`
  ).join("");
})();

// ---- Benchmark table ----
(function(){
  const rows = DATA.benchmark_names.map(name => {
    const s = DATA.benchmark_stats[name] || {};
    const qT=s.queries_total||0, qD=s.queries_done||0;
    const pT=s.passages_total||0, pD=s.passages_done||0;
    const tot=qT+pT, done=qD+pD;
    const pct=tot?Math.round(done*100/tot):0;
    const full = pct===100;
    return `<tr>
      <td><strong>${esc(name)}</strong></td>
      <td>
        <div class="bar-wrap"><div class="bar-fill ${full?'complete':''}" style="width:${pct}%"></div></div>
        <span class="pct">${pct}%</span>
      </td>
      <td>${qD}/${qT}</td>
      <td>${pD}/${pT}</td>
      <td><strong>${done}/${tot}</strong></td>
    </tr>`;
  });
  document.getElementById("bm-body").innerHTML = rows.join("");
})();

// ---- Annotator table ----
(function(){
  const rows = Object.entries(DATA.annotator_stats).map(([name, st]) => {
    const scores = st.scores || {};
    const total_submitted = [1,2,3,4,5].reduce((s,i)=>s+(scores[i]||0),0);
    const avg = total_submitted
      ? ([1,2,3,4,5].reduce((s,i)=>s+i*(scores[i]||0),0)/total_submitted).toFixed(2)
      : "—";

    // Distribution bar
    const segs = [1,2,3,4,5].map(i => {
      const n = scores[i]||0;
      const w = total_submitted ? Math.round(n*100/total_submitted) : 0;
      return w>0 ? `<div class="dist-seg" style="width:${w}%;background:${SCORE_HEX[i]}" title="${i}: ${n}"></div>` : "";
    }).join("");

    return `<tr>
      <td><strong>${esc(name)}</strong></td>
      <td><strong>${st.total}</strong></td>
      <td><div class="dist-bar">${segs}</div></td>
      <td>${avg}</td>
      ${[1,2,3,4,5].map(i=>`<td>${scores[i]||0}</td>`).join("")}
      <td>${st.discarded||0}</td>
    </tr>`;
  });
  document.getElementById("ann-body").innerHTML = rows.length
    ? rows.join("") : `<tr><td colspan="10" class="empty">No annotations yet.</td></tr>`;
})();

// ---- Annotations table ----
let sortCol = "date", sortDir = -1, expandedId = null;

// Populate filter dropdowns
(function(){
  const annotators = [...new Set(DATA.annotations.map(a=>a.annotator))].sort();
  const benchmarks = [...new Set(DATA.annotations.map(a=>a.benchmark))].sort();
  const fA = document.getElementById("f-annotator");
  const fB = document.getElementById("f-benchmark");
  annotators.forEach(a => { const o=document.createElement("option"); o.value=o.textContent=a; fA.appendChild(o); });
  benchmarks.forEach(b => { const o=document.createElement("option"); o.value=o.textContent=b; fB.appendChild(o); });
})();

function getFiltered() {
  const fA = document.getElementById("f-annotator").value;
  const fB = document.getElementById("f-benchmark").value;
  const fT = document.getElementById("f-type").value;
  const fS = document.getElementById("f-score").value;
  const fQ = document.getElementById("f-search").value.toLowerCase().trim();

  return DATA.annotations.filter(a => {
    if (fA && a.annotator !== fA) return false;
    if (fB && a.benchmark !== fB) return false;
    if (fT && a.record_type !== fT) return false;
    if (fS && a.score !== fS) return false;
    if (fQ && ![a.source_en, a.translation_sr, a.correction, a.comment, a.annotator, a.benchmark]
               .some(t => (t||"").toLowerCase().includes(fQ))) return false;
    return true;
  });
}

function sortData(arr) {
  return [...arr].sort((a,b) => {
    let av = a[sortCol]||"", bv = b[sortCol]||"";
    if (sortCol === "score") { av = parseInt(av)||99; bv = parseInt(bv)||99; }
    if (av < bv) return -sortDir;
    if (av > bv) return sortDir;
    return 0;
  });
}

function scoreLabel(score) {
  const map = {"1":"1 – Netačan","2":"2 – Greške","3":"3 – Adekvatan","4":"4 – Dobar","5":"5 – Odličan","-":"Discarded","?":"?"};
  return map[score] || score;
}

function renderTable() {
  const filtered = sortData(getFiltered());
  document.getElementById("ann-count").textContent =
    filtered.length + " of " + DATA.annotations.length + " annotation" + (DATA.annotations.length!==1?"s":"");

  const tbody = document.getElementById("ann-body2");
  const empty = document.getElementById("ann-empty");

  if (!filtered.length) {
    tbody.innerHTML = "";
    empty.style.display = "";
    return;
  }
  empty.style.display = "none";

  tbody.innerHTML = filtered.map((a, i) => {
    const isNoCorr = (a.correction||"").trim().toLowerCase() === "no corrections";
    const corrPreview = isNoCorr
      ? `<span class="no-corr">No corrections</span>`
      : `<span class="trunc">${esc(trunc(a.correction, 80))}</span>`;
    const rowId = "row-" + i;
    return `<tr class="ann-row" data-idx="${i}" onclick="toggleExpand(this, ${i}, ${JSON.stringify(filtered[i]).replace(/</g,'\\u003c')})">
      <td style="color:#94a3b8;font-size:12px">${i+1}</td>
      <td><strong>${esc(a.annotator)}</strong></td>
      <td style="font-size:12px">${esc(a.benchmark)}</td>
      <td><span class="type-badge type-${esc(a.record_type)}">${esc(a.record_type)}</span></td>
      <td><span class="badge ${SCORE_COLOR[a.score]||'score-q'}">${esc(scoreLabel(a.score))}</span></td>
      <td><span class="trunc">${esc(trunc(a.source_en, 100))}</span></td>
      <td>${corrPreview}</td>
      <td><span class="trunc">${esc(trunc(a.comment, 80))}</span></td>
      <td style="font-size:12px;color:#64748b;white-space:nowrap">${esc(a.date)}</td>
    </tr>
    <tr class="expand-row" id="${rowId}" style="display:none">
      <td colspan="9"></td>
    </tr>`;
  }).join("");
}

function toggleExpand(row, idx, ann) {
  const expandRow = row.nextElementSibling;
  const isOpen = expandRow.style.display !== "none";

  // Close all others
  document.querySelectorAll(".expand-row").forEach(r => r.style.display = "none");
  document.querySelectorAll(".ann-row").forEach(r => r.classList.remove("expanded"));

  if (isOpen) return;

  row.classList.add("expanded");
  expandRow.style.display = "";

  const isNoCorr = (ann.correction||"").trim().toLowerCase() === "no corrections";
  const corrClass = isNoCorr ? "text-box no-correction" : "text-box correction-made";
  expandRow.querySelector("td").innerHTML = `
    <div class="expand-content">
      <div class="expand-col">
        <h4>&#127468;&#127463; English Source</h4>
        <div class="text-box">${esc(ann.source_en)}</div>
      </div>
      <div class="expand-col">
        <h4>&#127480;&#127479; Serbian Translation (machine)</h4>
        <div class="text-box">${esc(ann.translation_sr)}</div>
      </div>
      <div class="expand-col">
        <h4>&#9999;&#65039; Annotator Correction</h4>
        <div class="${corrClass}">${esc(ann.correction||"(discarded — no correction entered)")}</div>
      </div>
      <div class="expand-col">
        <h4>&#128172; Comment</h4>
        <div class="text-box">${esc(ann.comment||"(no comment)")}</div>
      </div>
    </div>`;
}

// Sorting
document.querySelectorAll("th.sortable").forEach(th => {
  th.addEventListener("click", () => {
    const col = th.dataset.col;
    if (sortCol === col) sortDir *= -1; else { sortCol = col; sortDir = 1; }
    document.querySelectorAll("th.sortable").forEach(t => {
      t.classList.remove("sort-asc","sort-desc");
    });
    th.classList.add(sortDir===1?"sort-asc":"sort-desc");
    renderTable();
  });
});

// Filters
["f-annotator","f-benchmark","f-type","f-score","f-search"].forEach(id => {
  document.getElementById(id).addEventListener("input", renderTable);
});

// Initial render
renderTable();
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Generate a comprehensive HTML annotation report for NanoBEIR-sr"
    )
    parser.add_argument("--api-url", default=os.getenv("ARGILLA_API_URL"))
    parser.add_argument("--api-key", default=os.getenv("ARGILLA_API_KEY"))
    parser.add_argument("--workspace", default="argilla")
    parser.add_argument("--dataset-name", default="NanoBEIR-sr")
    parser.add_argument("--output", default="annotation_report.html",
                        help="Output HTML file path")
    parser.add_argument("--no-open", action="store_true",
                        help="Do not open the report in the browser after generating")
    args = parser.parse_args()

    if not args.api_url or not args.api_key:
        print("Error: --api-url and --api-key required (or set env vars)")
        sys.exit(1)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        client = rg.Argilla(api_url=args.api_url, api_key=args.api_key)

    dataset = client.datasets(name=args.dataset_name, workspace=args.workspace)
    if dataset is None:
        print(f"Error: dataset '{args.dataset_name}' not found.")
        sys.exit(1)

    print(f"Generating report for '{args.dataset_name}'...")
    data = fetch_data(client, dataset)

    total_done = data["total_done"]
    n_annotators = len(data["annotator_stats"])
    n_submitted = len(data["annotations"])
    print(f"\n  {total_done} records annotated by {n_annotators} annotator(s), "
          f"{n_submitted} total submissions")

    # Embed data safely — replace </ to prevent breaking the <script> tag
    data_json = json.dumps(data, ensure_ascii=False, default=str).replace("</", "<\\/")
    html = HTML_TEMPLATE.replace("__DATA_JSON__", data_json)

    out_path = Path(args.output)
    out_path.write_text(html, encoding="utf-8")
    print(f"\n  Report saved to: {out_path.resolve()}")

    if not args.no_open:
        webbrowser.open(out_path.resolve().as_uri())
        print("  Opening in browser...")


if __name__ == "__main__":
    main()
