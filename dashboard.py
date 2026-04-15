"""
Generate an interactive HTML analytics dashboard from Argilla annotations.

Pulls data from Argilla, analyzes it, and creates a standalone HTML file
with charts and per-annotator breakdowns.

Usage:
    python dashboard.py
    python dashboard.py --output report.html
"""

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime

import argilla as rg


def fetch_annotations(api_url: str, api_key: str, workspace: str,
                      dataset_name: str) -> dict:
    """Fetch all annotation data from Argilla and return structured stats."""

    client = rg.Argilla(api_url=api_url, api_key=api_key)
    dataset = client.datasets(name=dataset_name, workspace=workspace)

    if dataset is None:
        print(f"Error: dataset '{dataset_name}' not found.")
        sys.exit(1)

    # Build user ID -> username mapping
    users_by_id = {}
    try:
        for user in client.users:
            users_by_id[str(user.id)] = user.username
    except Exception:
        pass

    # Collect all data
    total_records = 0
    annotations = []  # list of individual annotation dicts

    no_correction_markers = ("bez ispravki", "no corrections", "")

    for record in dataset.records(with_responses=True):
        total_records += 1
        source_en = record.fields.get("source_text_en", "") if record.fields else ""
        translated_sr = record.fields.get("translated_text_sr", "") if record.fields else ""
        record_id = str(record.id)
        if record.metadata:
            record_id = record.metadata.get("task_id", record_id)

        if not record.responses:
            continue

        for response in record.responses:
            if response.status != "submitted":
                continue

            # Resolve annotator
            annotator = "unknown"
            if hasattr(response, "user_id") and response.user_id:
                uid = str(response.user_id)
                annotator = users_by_id.get(uid, uid)

            # Extract values - try multiple formats
            quality = None
            corrected = None

            # Argilla v2 stores responses differently
            # Try direct attribute access
            if hasattr(response, "values") and response.values:
                vals = response.values
                # Could be {"question_name": "value"} or {"question_name": {"value": "..."}}
                q = vals.get("quality_score")
                c = vals.get("corrected_text_sr")

                if isinstance(q, dict):
                    quality = q.get("value")
                elif isinstance(q, str):
                    quality = q

                if isinstance(c, dict):
                    corrected = c.get("value")
                elif isinstance(c, str):
                    corrected = c

            # Also try response.answers or similar
            if quality is None and hasattr(response, "answers"):
                ans = response.answers or {}
                quality = ans.get("quality_score")
                corrected = ans.get("corrected_text_sr")

            # Determine modification
            corrected_str = (corrected or "").strip()
            was_modified = corrected_str.lower() not in no_correction_markers

            annotations.append({
                "record_id": record_id,
                "annotator": annotator,
                "quality_score": quality,
                "corrected_text": corrected_str,
                "was_modified": was_modified,
                "source_en": source_en[:100],
                "translated_sr": translated_sr[:100],
            })

    return {
        "total_records": total_records,
        "annotations": annotations,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }


def compute_stats(data: dict) -> dict:
    """Compute aggregate statistics from raw data."""
    annotations = data["annotations"]
    total = data["total_records"]

    completed_ids = set(a["record_id"] for a in annotations)

    # Per annotator
    per_annotator = defaultdict(lambda: {"count": 0, "quality": Counter(), "modified": 0, "accepted": 0})
    quality_total = Counter()
    modified_total = 0
    accepted_total = 0

    for a in annotations:
        ann = a["annotator"]
        per_annotator[ann]["count"] += 1
        if a["quality_score"]:
            per_annotator[ann]["quality"][a["quality_score"]] += 1
            quality_total[a["quality_score"]] += 1
        if a["was_modified"]:
            per_annotator[ann]["modified"] += 1
            modified_total += 1
        else:
            per_annotator[ann]["accepted"] += 1
            accepted_total += 1

    return {
        "total_records": total,
        "completed_records": len(completed_ids),
        "total_annotations": len(annotations),
        "completion_pct": round(len(completed_ids) / total * 100, 1) if total else 0,
        "quality_distribution": dict(quality_total),
        "modified_total": modified_total,
        "accepted_total": accepted_total,
        "per_annotator": {k: dict(v) for k, v in per_annotator.items()},
        "sample_annotations": annotations[:20],
    }


def generate_html(stats: dict, timestamp: str) -> str:
    """Generate a standalone HTML dashboard."""

    quality_dist = stats["quality_distribution"]
    per_annotator = stats["per_annotator"]

    # Prepare chart data
    quality_labels = json.dumps(list(quality_dist.keys()) if quality_dist else ["no data"])
    quality_values = json.dumps(list(quality_dist.values()) if quality_dist else [0])
    quality_colors = json.dumps([
        {"low": "#ef4444", "medium": "#f59e0b", "high": "#22c55e"}.get(k, "#6b7280")
        for k in (quality_dist.keys() if quality_dist else ["no data"])
    ])

    annotator_names = json.dumps(list(per_annotator.keys()))
    annotator_counts = json.dumps([v["count"] for v in per_annotator.values()])

    # Per-annotator quality breakdown for stacked chart
    annotator_quality_data = {}
    for label in ["high", "medium", "low"]:
        annotator_quality_data[label] = [
            per_annotator[ann]["quality"].get(label, 0)
            for ann in per_annotator.keys()
        ]

    # Sample annotations table rows
    sample_rows = ""
    for a in stats["sample_annotations"]:
        q_class = {"low": "q-low", "medium": "q-med", "high": "q-high"}.get(a["quality_score"] or "", "")
        mod_icon = "&#9998;" if a["was_modified"] else "&#10003;"
        sample_rows += f"""
        <tr>
            <td>{a['annotator']}</td>
            <td><span class="{q_class}">{a['quality_score'] or 'N/A'}</span></td>
            <td>{mod_icon}</td>
            <td title="{a['source_en']}">{a['source_en'][:60]}...</td>
        </tr>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Translation Annotation Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
    * {{ margin: 0; padding: 0; box-sizing: border-box; }}
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #0f172a; color: #e2e8f0; padding: 24px; }}
    .header {{ text-align: center; margin-bottom: 32px; }}
    .header h1 {{ font-size: 24px; color: #f8fafc; margin-bottom: 4px; }}
    .header p {{ color: #94a3b8; font-size: 14px; }}
    .grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin-bottom: 24px; }}
    .card {{ background: #1e293b; border-radius: 12px; padding: 20px; }}
    .card-label {{ font-size: 12px; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.05em; }}
    .card-value {{ font-size: 32px; font-weight: 700; color: #f8fafc; margin-top: 4px; }}
    .card-sub {{ font-size: 13px; color: #64748b; margin-top: 2px; }}
    .charts {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 24px; }}
    .chart-box {{ background: #1e293b; border-radius: 12px; padding: 20px; }}
    .chart-box h3 {{ font-size: 14px; color: #94a3b8; margin-bottom: 12px; }}
    .full-width {{ grid-column: 1 / -1; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
    th {{ text-align: left; padding: 8px 12px; color: #94a3b8; border-bottom: 1px solid #334155; font-weight: 500; }}
    td {{ padding: 8px 12px; border-bottom: 1px solid #1e293b; }}
    .q-low {{ color: #ef4444; font-weight: 600; }}
    .q-med {{ color: #f59e0b; font-weight: 600; }}
    .q-high {{ color: #22c55e; font-weight: 600; }}
    .progress-bar {{ width: 100%; height: 8px; background: #334155; border-radius: 4px; overflow: hidden; margin-top: 8px; }}
    .progress-fill {{ height: 100%; background: linear-gradient(90deg, #3b82f6, #22c55e); border-radius: 4px; transition: width 0.5s; }}
    .annotator-table {{ margin-top: 8px; }}
    .annotator-table td {{ padding: 6px 12px; }}
</style>
</head>
<body>
<div class="header">
    <h1>Translation Annotation Dashboard</h1>
    <p>Serbian-AI-Society / translation-annotation-sr &mdash; Generated {timestamp}</p>
</div>

<div class="grid">
    <div class="card">
        <div class="card-label">Total Records</div>
        <div class="card-value">{stats['total_records']}</div>
    </div>
    <div class="card">
        <div class="card-label">Completed</div>
        <div class="card-value">{stats['completed_records']}</div>
        <div class="progress-bar"><div class="progress-fill" style="width:{stats['completion_pct']}%"></div></div>
        <div class="card-sub">{stats['completion_pct']}% complete</div>
    </div>
    <div class="card">
        <div class="card-label">Total Annotations</div>
        <div class="card-value">{stats['total_annotations']}</div>
        <div class="card-sub">across {len(per_annotator)} annotators</div>
    </div>
    <div class="card">
        <div class="card-label">Modification Rate</div>
        <div class="card-value">{round(stats['modified_total'] / max(stats['total_annotations'], 1) * 100)}%</div>
        <div class="card-sub">{stats['modified_total']} modified, {stats['accepted_total']} accepted</div>
    </div>
</div>

<div class="charts">
    <div class="chart-box">
        <h3>Quality Score Distribution</h3>
        <canvas id="qualityChart"></canvas>
    </div>
    <div class="chart-box">
        <h3>Annotations per Annotator</h3>
        <canvas id="annotatorChart"></canvas>
    </div>
    <div class="chart-box">
        <h3>Quality Scores by Annotator</h3>
        <canvas id="annotatorQualityChart"></canvas>
    </div>
    <div class="chart-box">
        <h3>Annotator Details</h3>
        <table class="annotator-table">
            <thead>
                <tr><th>Annotator</th><th>Done</th><th>High</th><th>Medium</th><th>Low</th><th>Modified</th></tr>
            </thead>
            <tbody>
                {"".join(f'''<tr>
                    <td>{ann}</td>
                    <td>{info['count']}</td>
                    <td class="q-high">{info['quality'].get('high', 0)}</td>
                    <td class="q-med">{info['quality'].get('medium', 0)}</td>
                    <td class="q-low">{info['quality'].get('low', 0)}</td>
                    <td>{info['modified']}/{info['count']}</td>
                </tr>''' for ann, info in per_annotator.items())}
            </tbody>
        </table>
    </div>
</div>

<div class="chart-box full-width" style="margin-bottom:24px;">
    <h3>Recent Annotations</h3>
    <table>
        <thead>
            <tr><th>Annotator</th><th>Quality</th><th>Modified</th><th>Source Text</th></tr>
        </thead>
        <tbody>
            {sample_rows}
        </tbody>
    </table>
</div>

<script>
const qualityCtx = document.getElementById('qualityChart').getContext('2d');
new Chart(qualityCtx, {{
    type: 'doughnut',
    data: {{
        labels: {quality_labels},
        datasets: [{{
            data: {quality_values},
            backgroundColor: {quality_colors},
            borderWidth: 0,
        }}]
    }},
    options: {{
        responsive: true,
        plugins: {{
            legend: {{ position: 'bottom', labels: {{ color: '#94a3b8' }} }}
        }}
    }}
}});

const annotatorCtx = document.getElementById('annotatorChart').getContext('2d');
new Chart(annotatorCtx, {{
    type: 'bar',
    data: {{
        labels: {annotator_names},
        datasets: [{{
            label: 'Annotations',
            data: {annotator_counts},
            backgroundColor: '#3b82f6',
            borderRadius: 6,
        }}]
    }},
    options: {{
        responsive: true,
        scales: {{
            y: {{ ticks: {{ color: '#94a3b8' }}, grid: {{ color: '#334155' }} }},
            x: {{ ticks: {{ color: '#94a3b8' }}, grid: {{ display: false }} }}
        }},
        plugins: {{ legend: {{ display: false }} }}
    }}
}});

const aqCtx = document.getElementById('annotatorQualityChart').getContext('2d');
new Chart(aqCtx, {{
    type: 'bar',
    data: {{
        labels: {annotator_names},
        datasets: [
            {{ label: 'High', data: {json.dumps(annotator_quality_data.get('high', []))}, backgroundColor: '#22c55e', borderRadius: 4 }},
            {{ label: 'Medium', data: {json.dumps(annotator_quality_data.get('medium', []))}, backgroundColor: '#f59e0b', borderRadius: 4 }},
            {{ label: 'Low', data: {json.dumps(annotator_quality_data.get('low', []))}, backgroundColor: '#ef4444', borderRadius: 4 }},
        ]
    }},
    options: {{
        responsive: true,
        scales: {{
            x: {{ stacked: true, ticks: {{ color: '#94a3b8' }}, grid: {{ display: false }} }},
            y: {{ stacked: true, ticks: {{ color: '#94a3b8' }}, grid: {{ color: '#334155' }} }}
        }},
        plugins: {{ legend: {{ position: 'bottom', labels: {{ color: '#94a3b8' }} }} }}
    }}
}});
</script>
</body>
</html>"""
    return html


def main():
    parser = argparse.ArgumentParser(description="Generate annotation dashboard")
    parser.add_argument("--output", default="dashboard.html", help="Output HTML file")
    parser.add_argument("--api-url", default=os.getenv("ARGILLA_API_URL"))
    parser.add_argument("--api-key", default=os.getenv("ARGILLA_API_KEY"))
    parser.add_argument("--workspace", default="argilla")
    parser.add_argument("--dataset-name", default="translation-annotation-sr")
    args = parser.parse_args()

    if not args.api_url or not args.api_key:
        print("Error: --api-url and --api-key required (or set env vars)")
        sys.exit(1)

    print("Fetching annotations from Argilla...")
    data = fetch_annotations(args.api_url, args.api_key, args.workspace, args.dataset_name)

    print("Computing statistics...")
    stats = compute_stats(data)

    print(f"Found {stats['total_annotations']} annotations from {len(stats['per_annotator'])} annotators.")

    # Debug: print raw quality data
    if not stats["quality_distribution"]:
        print("\nDEBUG: No quality scores found. Dumping first annotation response:")
        if data["annotations"]:
            print(json.dumps(data["annotations"][0], indent=2, ensure_ascii=False, default=str))

    html = generate_html(stats, data["timestamp"])

    with open(args.output, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"Dashboard saved to {args.output}")
    print(f"Open it in your browser: start {args.output}")


if __name__ == "__main__":
    main()