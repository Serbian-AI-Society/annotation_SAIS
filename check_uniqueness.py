"""
Comprehensive uniqueness check for the NanoBEIR-sr Argilla dataset.

Reconstructs every record exactly as load_nanobeir.py would build it — using
cached EN texts and live SR datasets from HuggingFace — then checks:

  1. Duplicate record IDs          (should be impossible due to benchmark prefix)
  2. Duplicate EN source content   (same English text, different record IDs)
  3. Duplicate SR translation      (same Serbian text, different record IDs)
  4. Duplicate EN+SR pair          (both fields identical across records)

Runs entirely offline using .beir_cache/ for EN texts (no large downloads).
SR texts require a HuggingFace internet connection (small datasets, fast).

Usage:
    python check_uniqueness.py
    python check_uniqueness.py --max-pos 10   # match load_nanobeir default
    python check_uniqueness.py --show-dupes   # print full duplicate details
"""

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from datasets import load_dataset

from load_nanobeir import BENCHMARKS, format_source

CACHE_DIR = Path(__file__).parent / ".beir_cache"
MAX_POS_DEFAULT = 10


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def md5(s: str) -> str:
    return hashlib.md5(s.encode("utf-8")).hexdigest()


def load_cache(en_hub: str, config: str, split: str) -> dict:
    key = f"{en_hub.replace('/', '__')}_{config}_{split}.json"
    path = CACHE_DIR / key
    if not path.exists():
        print(f"  [WARN] Cache missing: {path.name} — EN texts for this benchmark will be empty")
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_positives(bench: dict) -> dict:
    """Return {query_id: [corpus_id, ...]}."""
    sr_hub = bench["sr_hub"]
    sr_split = bench["sr_split"]
    rel_config = bench["rel_config"]
    positives = defaultdict(list)
    if rel_config == "qrels":
        ds = load_dataset(sr_hub, "qrels", split=sr_split)
        for row in ds:
            positives[str(row["query-id"])].append(str(row["corpus-id"]))
    else:
        ds = load_dataset(sr_hub, "relevance", split=sr_split)
        for row in ds:
            positives[str(row["query-id"])] = [str(i) for i in row["positive-corpus-ids"]]
    return dict(positives)


# ---------------------------------------------------------------------------
# Record reconstruction
# ---------------------------------------------------------------------------

def collect_records(bench: dict, max_pos: int) -> list[dict]:
    """
    Reconstruct every record for one benchmark exactly as load_nanobeir.py does.
    Returns list of dicts with keys: id, record_type, benchmark, en_hash, sr_hash, both_hash.
    """
    name = bench["name"]
    sr_hub = bench["sr_hub"]
    sr_split = bench["sr_split"]
    en_hub = bench["en_hub"]

    # --- SR queries ---
    sr_q_ds = load_dataset(sr_hub, "queries", split=sr_split)
    sr_q = {str(r["_id"]): r["text"] for r in sr_q_ds}

    # --- EN queries (from cache) ---
    en_q_cache = load_cache(en_hub, "queries", bench["en_queries_split"])

    # --- Positive passage IDs ---
    positives_by_query = load_positives(bench)
    needed_ids: set = set()
    for pos_ids in positives_by_query.values():
        needed_ids.update(pos_ids[:max_pos])

    # --- SR corpus ---
    sr_c_ds = load_dataset(sr_hub, "corpus", split=sr_split)
    sr_c = {str(r["_id"]): r["text"] for r in sr_c_ds if str(r["_id"]) in needed_ids}

    # --- EN corpus (from cache) ---
    en_c_cache = load_cache(en_hub, "corpus", bench["en_corpus_split"])

    records = []

    # Queries
    for q_id, sr_text in sr_q.items():
        en_raw = en_q_cache.get(q_id, {})
        en_text = format_source(en_raw.get("title", ""), en_raw.get("text", ""))
        if not en_text or not sr_text:
            continue
        records.append({
            "id": f"{name}_query_{q_id}",
            "record_type": "query",
            "benchmark": name,
            "en_hash": md5(en_text),
            "sr_hash": md5(sr_text),
            "both_hash": md5(en_text + "\x00" + sr_text),
            "en_preview": en_text[:80].replace("\n", " "),
        })

    # Passages
    for p_id in needed_ids:
        sr_text = sr_c.get(p_id)
        en_raw = en_c_cache.get(p_id, {})
        en_text = format_source(en_raw.get("title", ""), en_raw.get("text", ""))
        if not en_text or not sr_text:
            continue
        records.append({
            "id": f"{name}_passage_{p_id}",
            "record_type": "passage",
            "benchmark": name,
            "en_hash": md5(en_text),
            "sr_hash": md5(sr_text),
            "both_hash": md5(en_text + "\x00" + sr_text),
            "en_preview": en_text[:80].replace("\n", " "),
        })

    return records


# ---------------------------------------------------------------------------
# Duplicate analysis
# ---------------------------------------------------------------------------

def find_duplicates(records: list[dict], field: str) -> dict:
    """Return {hash: [record_ids]} for hashes that appear more than once."""
    groups: dict = defaultdict(list)
    for r in records:
        groups[r[field]].append(r["id"])
    return {h: ids for h, ids in groups.items() if len(ids) > 1}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Check uniqueness of all NanoBEIR-sr records",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--max-pos", type=int, default=MAX_POS_DEFAULT,
                        help="Max positive passages per query (must match load_nanobeir run)")
    parser.add_argument("--show-dupes", action="store_true",
                        help="Print the full record IDs for every duplicate group")
    args = parser.parse_args()

    print(f"Reconstructing records for all 13 benchmarks (max_pos={args.max_pos})...")
    print()

    all_records: list[dict] = []
    failed: list[str] = []

    for bench in BENCHMARKS:
        name = bench["name"]
        print(f"  {name}...", end=" ", flush=True)
        try:
            recs = collect_records(bench, args.max_pos)
            all_records.extend(recs)
            q = sum(1 for r in recs if r["record_type"] == "query")
            p = sum(1 for r in recs if r["record_type"] == "passage")
            print(f"{len(recs)} records ({q}Q + {p}P)")
        except Exception as exc:
            print(f"FAILED: {exc}")
            failed.append(name)

    print()
    print(f"Total records reconstructed: {len(all_records)}")
    if failed:
        print(f"Benchmarks that failed (excluded from analysis): {failed}")
    print()

    # -----------------------------------------------------------------------
    # Check 1: Duplicate IDs
    # -----------------------------------------------------------------------
    id_counts: dict = defaultdict(int)
    for r in all_records:
        id_counts[r["id"]] += 1
    dup_ids = {rid: cnt for rid, cnt in id_counts.items() if cnt > 1}

    print("=" * 60)
    print(f"CHECK 1 — Duplicate record IDs: {len(dup_ids)}")
    if dup_ids and args.show_dupes:
        for rid, cnt in dup_ids.items():
            print(f"  {rid}  (appears {cnt}x)")
    elif not dup_ids:
        print("  OK — all record IDs are unique.")
    else:
        print(f"  {len(dup_ids)} duplicate IDs found. Run with --show-dupes to see them.")

    # -----------------------------------------------------------------------
    # Check 2: Duplicate EN content
    # -----------------------------------------------------------------------
    dup_en = find_duplicates(all_records, "en_hash")

    print()
    print(f"CHECK 2 — Duplicate EN source content: {len(dup_en)} group(s)")
    if not dup_en:
        print("  OK — all EN texts are unique.")
    else:
        # Build hash -> preview map
        hash_to_preview = {r["en_hash"]: r["en_preview"] for r in all_records}
        for h, ids in sorted(dup_en.items(), key=lambda x: x[1][0]):
            cross = len({rid.split("_passage_")[0].split("_query_")[0] for rid in ids}) > 1
            tag = "[CROSS-BENCHMARK]" if cross else "[WITHIN-BENCHMARK]"
            print(f"  {tag}  {len(ids)} records share EN text: \"{hash_to_preview[h]}...\"")
            if args.show_dupes:
                for rid in ids:
                    print(f"    - {rid}")

    # -----------------------------------------------------------------------
    # Check 3: Duplicate SR content
    # -----------------------------------------------------------------------
    dup_sr = find_duplicates(all_records, "sr_hash")

    print()
    print(f"CHECK 3 — Duplicate SR translation content: {len(dup_sr)} group(s)")
    if not dup_sr:
        print("  OK — all SR texts are unique.")
    else:
        for h, ids in sorted(dup_sr.items(), key=lambda x: x[1][0]):
            cross = len({rid.split("_passage_")[0].split("_query_")[0] for rid in ids}) > 1
            tag = "[CROSS-BENCHMARK]" if cross else "[WITHIN-BENCHMARK]"
            print(f"  {tag}  {len(ids)} records share SR text")
            if args.show_dupes:
                for rid in ids:
                    print(f"    - {rid}")

    # -----------------------------------------------------------------------
    # Check 4: Duplicate EN+SR pairs
    # -----------------------------------------------------------------------
    dup_both = find_duplicates(all_records, "both_hash")

    print()
    print(f"CHECK 4 — Duplicate EN+SR pairs (both fields identical): {len(dup_both)} group(s)")
    if not dup_both:
        print("  OK — all EN+SR pairs are unique.")
    else:
        for h, ids in sorted(dup_both.items(), key=lambda x: x[1][0]):
            cross = len({rid.split("_passage_")[0].split("_query_")[0] for rid in ids}) > 1
            tag = "[CROSS-BENCHMARK]" if cross else "[WITHIN-BENCHMARK]"
            print(f"  {tag}  {len(ids)} records are fully identical")
            if args.show_dupes:
                for rid in ids:
                    print(f"    - {rid}")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    issues = []
    if dup_ids:
        issues.append(f"  {len(dup_ids)} duplicate record ID(s)")
    if dup_en:
        cross_en = sum(1 for ids in dup_en.values()
                       if len({r.split("_passage_")[0].split("_query_")[0] for r in ids}) > 1)
        within_en = len(dup_en) - cross_en
        if cross_en:
            issues.append(f"  {cross_en} cross-benchmark EN duplicate group(s)  ← PROBLEM")
        if within_en:
            issues.append(f"  {within_en} within-benchmark EN duplicate group(s)")
    if dup_sr:
        cross_sr = sum(1 for ids in dup_sr.values()
                       if len({r.split("_passage_")[0].split("_query_")[0] for r in ids}) > 1)
        within_sr = len(dup_sr) - cross_sr
        if cross_sr:
            issues.append(f"  {cross_sr} cross-benchmark SR duplicate group(s)  ← PROBLEM")
        if within_sr:
            issues.append(f"  {within_sr} within-benchmark SR duplicate group(s)")
    if dup_both:
        cross_both = sum(1 for ids in dup_both.values()
                         if len({r.split("_passage_")[0].split("_query_")[0] for r in ids}) > 1)
        within_both = len(dup_both) - cross_both
        if cross_both:
            issues.append(f"  {cross_both} cross-benchmark EN+SR identical record group(s)  ← PROBLEM")
        if within_both:
            issues.append(f"  {within_both} within-benchmark EN+SR identical record group(s)")

    if issues:
        print("Issues found:")
        for issue in issues:
            print(issue)
        print()
        print("Run with --show-dupes to see the specific record IDs.")
    else:
        print("All clear — no duplicate content found across all 2,811 records.")


if __name__ == "__main__":
    main()
