# NanoBEIR-sr Annotation Pipeline — Claude Context

## What this project is

Human annotation pipeline for Serbian machine translations of NanoBEIR benchmarks.
13 benchmarks, 2,811 records. Translations by DeepSeek-V3. Hosted on Argilla (HuggingFace Space).

**Argilla Space:** https://serbian-ai-society-argilla-annotation.hf.space  
**Owner API key:** ask the user or check HuggingFace Space Settings → Secrets → PASSWORD.

## Architecture

Two Argilla datasets:
- `NanoBEIR-sr` — main annotation, `TaskDistribution(min_submitted=1)` (= one annotator per record)
- `NanoBEIR-sr-calibration` — IAA calibration, `OverlapTaskDistribution(min_submitted=100)` (every annotator does all 26 records; records never auto-complete)

Both share the same schema defined in `load_nanobeir.py:build_settings()`:
- **Fields (left panel):** `annotation_guide_link` (markdown, hyperlink to guide), `source_text_en`, `translated_text_sr`
- **Questions (right panel):** `quality_score` (LabelQuestion 1–5), `comment` (TextQuestion), `corrected_text_sr` (TextQuestion, pre-filled via `rg.Suggestion`)

The correction field (`corrected_text_sr`) is pre-filled with the machine translation via `rg.Suggestion(question_name="corrected_text_sr", value=sr_text, agent="DeepSeek-V3", type="model")`. Annotators edit in-place.

## Key technical constraints

- **Argilla v2**: Fields are immutable after dataset creation. Schema changes require `--recreate` (deletes all annotations).
- `TaskDistribution` is an alias for `OverlapTaskDistribution` in this Argilla version — same class.
- `rg.TaskDistribution(min_submitted=1)` maps to `OverlapTaskDistributionModel(strategy='overlap', min_submitted=1)`.
- `client.http_client` is a plain `httpx.Client` — use it for direct API calls (e.g., `DELETE /api/v1/responses/{id}`).
- Record statuses: `pending` → `completed` (after min_submitted responses) or stays pending forever with min_submitted=100.
- Discarding a response removes it from the annotator's queue but does NOT count toward min_submitted.

## Scripts — what each does

| Script | Purpose |
|--------|---------|
| `load_nanobeir.py` | Upload all 13 benchmarks to `NanoBEIR-sr`. `--recreate` to rebuild. |
| `create_calibration_set.py` | Sample 26 records and create `NanoBEIR-sr-calibration`. Always use `--min-submitted 100`. |
| `manage_annotators.py` | Add/remove annotators. They must log in at Space URL first. |
| `check_progress.py` | Terminal dashboard: per-benchmark progress + per-annotator stats. |
| `generate_report.py` | Self-contained HTML report. Opens in browser. |
| `fix_calibration_discards.py` | **Recovery tool**: deletes discarded responses so records reappear in annotator queues. Dry run by default; add `--fix` to apply. |
| `compute_agreement.py` | Cohen's kappa for all annotator pairs in the calibration dataset. Queries ALL records (not just completed) because min_submitted=100 means records never complete. |
| `export_annotations_V2.py` | Export completed annotations to HF Hub / JSONL. |
| `check_uniqueness.py` | Offline duplicate detection across all 2,811 records. |

## Known issues and their fixes

### Calibration records discarded by annotator
**Symptom:** Annotator's calibration queue is shorter than 26.
**Cause:** They clicked Discard instead of Submit.
**Fix:** `python fix_calibration_discards.py --fix` — deletes discarded responses via `DELETE /api/v1/responses/{id}`, restoring records to pending for that annotator.

### Phantom corrections
**Symptom:** A correction value equals the original machine translation (annotator submitted without editing the pre-filled text).
**Fix:** Both `export_annotations_V2.py` and `generate_report.py` detect `corrected_text_sr == translated_text_sr` and treat it as "no correction made".

### Dataset schema change needed
**Symptom:** Need to add/remove a field.
**Fix:** `python load_nanobeir.py --recreate` — deletes dataset and all annotations, then rebuilds. Confirm with user first if real annotations exist.

### `compute_agreement.py` returning no results
Was previously querying `status == "completed"` but calibration records never complete. Fixed: now queries all records.

## Calibration workflow

1. All annotators added to workspace via `manage_annotators.py add`.
2. Each annotator completes all 26 calibration records (warn them: do not use Discard).
3. If anyone has fewer than 26 in queue: run `fix_calibration_discards.py --fix`.
4. Run `compute_agreement.py` — target pairwise Cohen's kappa ≥ 0.6.
5. If agreement is good, proceed with main dataset.

## Credentials pattern

All scripts accept `--api-url` / `--api-key` or read from `ARGILLA_API_URL` / `ARGILLA_API_KEY` env vars. For one-off commands:

```bash
ARGILLA_API_URL=https://serbian-ai-society-argilla-annotation.hf.space \
ARGILLA_API_KEY=<key> python <script>.py
```
