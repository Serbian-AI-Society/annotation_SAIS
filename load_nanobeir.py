"""
Load all 13 NanoBEIR benchmarks into a single merged Argilla dataset for translation annotation.

All 2,811 records (queries + positive passages from all 13 benchmarks) are collected,
shuffled randomly, and uploaded to one Argilla dataset. This ensures annotators see a
balanced mix from all benchmarks rather than working through one benchmark at a time.

Records are identified by benchmark-prefixed IDs (e.g. NanoArguAna_query_42) so records
from different benchmarks never collide, and --only reruns safely upsert into the existing
merged dataset.

For each benchmark the script:
  - Loads all queries with EN source paired from BeIR + SR translation
  - Loads positive passages (capped at --max-pos per query, default 10)
    to keep annotation scope manageable for high-positive datasets
    (NanoNFCorpus avg 50/query, NanoDBPedia avg 23/query, NanoTouche2020 avg 19/query)

English source is fetched via streaming for large BeIR corpora (up to 8.8M rows) and
cached to .beir_cache/ — subsequent runs (after Space restarts) load from disk instantly.
NanoClimateFEVER is sourced from NanoBEIR-sr (no individual bm25 dataset exists).

If one benchmark fails, the others continue. Use --only to rerun / add a single benchmark.

Usage:
    # Full run — creates/updates merged dataset with all 13 benchmarks
    python load_nanobeir.py

    # Only add/refresh one benchmark in the merged dataset
    python load_nanobeir.py --only NanoArguAna

    # Dry run (prints counts, no upload)
    python load_nanobeir.py --dry-run

    # Custom dataset name or passage cap
    python load_nanobeir.py --dataset-name my-annotation-dataset --max-pos 5

    # Upload without shuffling (preserves benchmark-then-record-type order)
    python load_nanobeir.py --no-shuffle

Environment variables (or pass as args):
    ARGILLA_API_URL
    ARGILLA_API_KEY
"""

import os
import sys
import json
import argparse
import logging
import warnings
import urllib.request
import urllib.parse
from pathlib import Path
from collections import defaultdict

# Must be set before importing datasets to avoid DLL block on Windows
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import argilla as rg
from datasets import load_dataset

# Directory for caching EN text lookups from large BeIR corpora.
# Populated on first run; subsequent runs load from disk instantly.
CACHE_DIR = Path(__file__).parent / ".beir_cache"

# Corpora too large for row-by-row streaming (IDs scattered throughout millions of rows).
# For these we download the corpus Parquet file once and filter locally with pyarrow
# row-group statistics. Results are cached to .beir_cache/ so subsequent runs load instantly.
API_FETCH_CORPORA = {
    "BeIR/msmarco",
    "BeIR/hotpotqa",
    "BeIR/fever",
    "BeIR/climate-fever",
    "BeIR/dbpedia-entity",
}

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = open(sys.stdout.fileno(), mode="w", encoding="utf-8", buffering=1)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Annotation guidelines
# ---------------------------------------------------------------------------
GUIDE_URL = "https://docs.google.com/document/d/1XDhdm9bF8habXG9xE-_WlXXy6-BLee1m2gLqQGrU8Mg/edit?tab=t.0#heading=h.vhgcdvffaq5m"
GUIDE_LINK_FIELD = f"[Ovde se možete vratiti na uputstva za anotaciju]({GUIDE_URL})"

GUIDELINES = """
# Uputstvo za ispravljanje automatskog prevoda (NanoBEIR)

## Opis podataka

Skupovi podataka su podskupovi NanoBEIR benchmark kolekcije — standardnog evaluacionog skupa za srpske modele za pretraživanje informacija. Originalni tekstovi su na engleskom. Automatski prevod sa engleskog na srpski urađen je korišćenjem DeepSeek-V3.

Vaš zadatak je da proverite i ispravite kvalitet automatskog prevoda upita (queries) i pasusa (passages) iz ovih skupova podataka.

## Pravila

- Ispravljajte svaki primer zasebno i nezavisno od drugih primera.
- Ispravku primera radite samostalno bez korišćenja LLM-ova (ChatGPT, Gemini, Claude, i sl.).
- Ne koristite LLM-ove za proveru ispravki teksta koje ste uneli.
- Za proveru tačnosti prevoda imenovanih entiteta i stručne terminologije koristite Google pretragu.

## Proces anotacije

### Korak 1: Procena ispravnosti teksta (Engleski original naspram srpskog prevoda)

Prvo pročitajte tekst na engleskom, zatim proverite prevod na srpskom. Proverite da li:
- Tekst na srpskom ima isto značenje kao tekst na engleskom
- Informacije iz teksta na engleskom nisu izostavljene u srpskom
- U srpskom prevodu ne postoji dodat tekst koji ne postoji u originalnom engleskom tekstu
- Namera, ton, i semantičko značenje originalnog teksta je isto i u prevedenom tekstu

### Korak 2: Kvalitet jezika (Samo srpski prevod)

Ponovo pročitajte samo tekst na srpskom, i proverite:
- Da li je prevod u skladu sa pravilima srpske gramatike, sintakse, i leksike
- Da li su glagoli, imenice i zamenice u ispravnom rodu, broju i padežu
- Da li su imenovani entiteti (lična imena, geografske lokacije, institucije) prevedeni tačno
- Da li u prevodu postoje pravopisne greške

### Korak 3: Unos ocene, vrsta grešaka, ispravki i komentara

**Ocena kvaliteta (1–5):**

- 1 – Potpuno netačan: Prevod ne prenosi značenje originalnog teksta.
- 2 – Veće greške: Prevod menja značenje ili ključna terminologija je pogrešno prevedena.
- 3 – Adekvatan: Tačno prenosi značenje, ali sadrži greške koje mogu uticati na razumevanje.
- 4 – Dobar: Potpuno i tačno prenosi značenje. Sadrži manje stilske greške koje ne utiču na razumevanje.
- 5 – Odličan: U potpunosti prenosi značenje. Prirodan srpski jezik, bez gramatičkih ili stilskih grešaka.

**Vrste grešaka:** Označite sve vrste grešaka koje ste pronašli. Ako prevod nema grešaka, označite samo „Nema grešaka" na dnu liste.

| Kategorija | Opis |
|---|---|
| Pogrešno značenje | Prevod prenosi drugačije značenje od originalnog teksta |
| Nedostaje informacija | Deo sadržaja iz originalnog teksta nije preveden |
| Dodat sadržaj | U prevodu postoji sadržaj koji ne postoji u originalu |
| Gramatičke greške | Greške u sintaksi ili strukturi rečenice |
| Deklinacija / konjugacija | Pogrešan rod, broj, padež ili lice glagola |
| Imenovani entiteti | Pogrešno prevedeno ime osobe, mesta, institucije ili organizacije |
| Pravopis | Pravopisne greške |
| Terminologija | Pogrešno prevedeni stručni termini |
| Neprirodan stil | Sintaktički ispravno, ali zvuči kao mašinski prevod — neprirodni srpski |
| Nema grešaka | Prevod je ispravan i prirodan — nema potrebe za ispravkama |

**Ispravke:** Srpski prevod je unapred učitan u polje za ispravke. Ispravite ga direktno. Ako prevodu nisu potrebne ispravke, obrišite tekst i unesite tačno: No corrections.

**Komentar (opciono):** Napišite komentar samo ako vrste grešaka ne opisuju problem dovoljno.
"""

# ---------------------------------------------------------------------------
# Benchmark configuration
# All 13 NanoBEIR sub-datasets with their English BeIR sources.
# Split names confirmed: all BeIR datasets use 'corpus'/'queries' as split names.
# NanoClimateFEVER uses NanoBEIR-sr (no individual bm25 dataset) with qrels config
# (flat one-row-per-pair) instead of the 'relevance' config (list-per-query).
# ---------------------------------------------------------------------------
BENCHMARKS = [
    {
        "name": "NanoArguAna",
        "sr_hub": "Serbian-AI-Society/NanoArguAna-bm25",
        "sr_split": "train",
        "rel_config": "relevance",
        "en_hub": "BeIR/arguana",
        "en_corpus_split": "corpus",
        "en_queries_split": "queries",
    },
    {
        "name": "NanoTouche2020",
        "sr_hub": "Serbian-AI-Society/NanoTouche2020-bm25",
        "sr_split": "train",
        "rel_config": "relevance",
        "en_hub": "BeIR/webis-touche2020",
        "en_corpus_split": "corpus",
        "en_queries_split": "queries",
    },
    {
        "name": "NanoSciFact",
        "sr_hub": "Serbian-AI-Society/NanoSciFact-bm25",
        "sr_split": "train",
        "rel_config": "relevance",
        "en_hub": "BeIR/scifact",
        "en_corpus_split": "corpus",
        "en_queries_split": "queries",
    },
    {
        "name": "NanoSCIDOCS",
        "sr_hub": "Serbian-AI-Society/NanoSCIDOCS-bm25",
        "sr_split": "train",
        "rel_config": "relevance",
        "en_hub": "BeIR/scidocs",
        "en_corpus_split": "corpus",
        "en_queries_split": "queries",
    },
    {
        "name": "NanoNQ",
        "sr_hub": "Serbian-AI-Society/NanoNQ-bm25",
        "sr_split": "train",
        "rel_config": "relevance",
        "en_hub": "BeIR/nq",
        "en_corpus_split": "corpus",
        "en_queries_split": "queries",
    },
    {
        "name": "NanoNFCorpus",
        "sr_hub": "Serbian-AI-Society/NanoNFCorpus-bm25",
        "sr_split": "train",
        "rel_config": "relevance",
        "en_hub": "BeIR/nfcorpus",
        "en_corpus_split": "corpus",
        "en_queries_split": "queries",
    },
    {
        "name": "NanoMSMARCO",
        "sr_hub": "Serbian-AI-Society/NanoMSMARCO-bm25",
        "sr_split": "train",
        "rel_config": "relevance",
        "en_hub": "BeIR/msmarco",
        "en_corpus_split": "corpus",
        "en_queries_split": "queries",
    },
    {
        "name": "NanoFiQA2018",
        "sr_hub": "Serbian-AI-Society/NanoFiQA2018-bm25",
        "sr_split": "train",
        "rel_config": "relevance",
        "en_hub": "BeIR/fiqa",
        "en_corpus_split": "corpus",
        "en_queries_split": "queries",
    },
    {
        "name": "NanoHotpotQA",
        "sr_hub": "Serbian-AI-Society/NanoHotpotQA-bm25",
        "sr_split": "train",
        "rel_config": "relevance",
        "en_hub": "BeIR/hotpotqa",
        "en_corpus_split": "corpus",
        "en_queries_split": "queries",
    },
    {
        "name": "NanoFEVER",
        "sr_hub": "Serbian-AI-Society/NanoFEVER-bm25",
        "sr_split": "train",
        "rel_config": "relevance",
        "en_hub": "BeIR/fever",
        "en_corpus_split": "corpus",
        "en_queries_split": "queries",
    },
    {
        "name": "NanoDBPedia",
        "sr_hub": "Serbian-AI-Society/NanoDBPedia-bm25",
        "sr_split": "train",
        "rel_config": "relevance",
        "en_hub": "BeIR/dbpedia-entity",
        "en_corpus_split": "corpus",
        "en_queries_split": "queries",
    },
    {
        "name": "NanoQuoraRetrieval",
        "sr_hub": "Serbian-AI-Society/NanoQuoraRetrieval-bm25",
        "sr_split": "train",
        "rel_config": "relevance",
        "en_hub": "BeIR/quora",
        "en_corpus_split": "corpus",
        "en_queries_split": "queries",
    },
    {
        # Only exists as a split inside NanoBEIR-sr (no individual bm25 dataset).
        # Uses 'qrels' config (flat: one row per query-corpus pair) instead of
        # 'relevance' (one row per query with a list of positives).
        "name": "NanoClimateFEVER",
        "sr_hub": "Serbian-AI-Society/NanoBEIR-sr",
        "sr_split": "NanoClimateFEVER",
        "rel_config": "qrels",
        "en_hub": "BeIR/climate-fever",
        "en_corpus_split": "corpus",
        "en_queries_split": "queries",
    },
]


# ---------------------------------------------------------------------------
# Argilla dataset settings (same schema for all 13 benchmarks)
# ---------------------------------------------------------------------------
def build_settings(distribution=None) -> rg.Settings:
    """
    Build Argilla dataset settings (shared schema for all datasets).

    distribution: pass rg.OverlapTaskDistribution(min_submitted=N) for a
    calibration dataset where every annotator must annotate every record.
    Defaults to rg.TaskDistribution(min_submitted=1) — one annotator per record.
    """
    if distribution is None:
        distribution = rg.TaskDistribution(min_submitted=1)
    return rg.Settings(
        guidelines=GUIDELINES,
        fields=[
            rg.TextField(
                name="annotation_guide_link",
                title="📋 Uputstvo",
                use_markdown=True,
                required=False,
            ),
            rg.TextField(
                name="source_text_en",
                title="🇬🇧 English Source Text (Originalni tekst na engleskom)",
                use_markdown=False,
                required=True,
            ),
            rg.TextField(
                name="translated_text_sr",
                title="🇷🇸 Machine Translation (Mašinski prevod na srpski)",
                use_markdown=False,
                required=True,
            ),
        ],
        questions=[
            rg.LabelQuestion(
                name="quality_score",
                title="Ocena kvaliteta prevoda (izaberite jednu ocenu)",
                description=(
                    "Detaljan opis svake ocene dostupan je u smernicama (Guidelines)."
                ),
                labels=[
                    "1 – Potpuno netačan",
                    "2 – Veće greške",
                    "3 – Adekvatan",
                    "4 – Dobar",
                    "5 – Odličan",
                ],
                required=True,
            ),
            rg.MultiLabelQuestion(
                name="error_categories",
                title="Vrste grešaka (označite sve koje se odnose na prevod)",
                description=(
                    "Označite sve vrste grešaka koje ste pronašli. "
                    "Ako nema grešaka, označite 'Nema grešaka' na dnu liste."
                ),
                labels=[
                    "Pogrešno značenje",
                    "Nedostaje informacija",
                    "Dodat sadržaj",
                    "Gramatičke greške",
                    "Deklinacija / konjugacija",
                    "Imenovani entiteti",
                    "Pravopis",
                    "Terminologija",
                    "Neprirodan stil",
                    "Nema grešaka",
                ],
                required=True,
            ),
            rg.TextQuestion(
                name="corrected_text_sr",
                title="🇷🇸 Ispravite mašinski prevod na srpski",
                description=(
                    "Prevod je unapred učitan. Ispravite ga direktno. "
                    "Ako prevod ne zahteva ispravke, obrišite tekst i unesite: No corrections."
                ),
                required=True,
                use_markdown=False,
            ),
            rg.TextQuestion(
                name="comment",
                title="Komentar (opciono)",
                description=(
                    "Opciono: napišite komentar ako vrste grešaka ne opisuju problem dovoljno."
                ),
                required=False,
                use_markdown=False,
            ),
        ],
        metadata=[
            rg.TermsMetadataProperty(
                name="task_id",
                title="Task ID",
                visible_for_annotators=True,
            ),
            rg.TermsMetadataProperty(
                name="record_type",
                title="Record Type (query / passage)",
                visible_for_annotators=True,
            ),
            rg.TermsMetadataProperty(
                name="benchmark",
                title="Benchmark",
                visible_for_annotators=True,
            ),
        ],
        allow_extra_metadata=True,
        distribution=distribution,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def get_or_create_dataset(
    client: rg.Argilla, argilla_name: str, workspace: str
) -> rg.Dataset:
    """Return existing Argilla dataset or create a new one."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        existing = client.datasets(name=argilla_name, workspace=workspace)
    if existing is not None:
        log.info(f"  Dataset '{argilla_name}' already exists — will append/update records.")
        return existing

    dataset = rg.Dataset(
        name=argilla_name,
        workspace=workspace,
        settings=build_settings(),
        client=client,
    )
    dataset.create()
    log.info(f"  Created dataset '{argilla_name}'.")
    return dataset


def format_source(title: str, text: str) -> str:
    """
    Combine BeIR title and text for display in Argilla.
    Some BeIR datasets include a descriptive title (e.g. ArguAna topic tags);
    prepend it so annotators have full context.
    """
    title = (title or "").strip()
    text = (text or "").strip()
    if title:
        return f"{title}\n\n{text}"
    return text


def _cache_path(en_hub: str, config: str, split: str) -> Path:
    """Return the cache file path for a given BeIR dataset/config/split."""
    key = f"{en_hub.replace('/', '__')}_{config}_{split}.json"
    return CACHE_DIR / key


def _load_cache(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _save_cache(path: Path, data: dict) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)


def _stream_fetch(en_hub: str, config: str, split: str, ids_set: set) -> dict:
    """Row-by-row streaming fetch. Good when target IDs appear early in the file."""
    log.info(
        f"    Streaming {len(ids_set)} IDs from {en_hub} [{config}/{split}]"
        f" (early-stop when all found)..."
    )
    result: dict = {}
    scanned = 0
    ds = load_dataset(en_hub, config, split=split, streaming=True)
    for row in ds:
        scanned += 1
        rid = str(row["_id"])
        if rid in ids_set:
            result[rid] = {
                "title": row.get("title", "") or "",
                "text": row.get("text", "") or "",
            }
            if len(result) == len(ids_set):
                break
        if scanned % 500_000 == 0:
            log.info(f"    ...scanned {scanned:,} rows, found {len(result)}/{len(ids_set)}")
    found = len(result)
    missing = len(ids_set) - found
    msg = f"    Found {found}/{len(ids_set)} EN texts (scanned {scanned:,} rows)"
    if missing:
        msg += f" — {missing} IDs not found in BeIR source"
    log.info(msg)
    return result


def _get_parquet_url(en_hub: str, config: str, split: str) -> str | None:
    """Return the first Parquet file URL for a BeIR dataset split."""
    api_url = (
        "https://datasets-server.huggingface.co/parquet"
        f"?dataset={urllib.parse.quote(en_hub)}"
    )
    try:
        req = urllib.request.Request(api_url, headers={"User-Agent": "load_nanobeir/1.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        files = [
            f for f in data.get("parquet_files", [])
            if f["config"] == config and f["split"] == split
        ]
        return files[0]["url"] if files else None
    except Exception as exc:
        log.warning(f"    Could not get Parquet URL for {en_hub} [{config}/{split}]: {exc}")
        return None


def _smart_parquet_fetch(en_hub: str, config: str, split: str, ids_set: set) -> dict:
    """
    Download the corpus Parquet file and filter for target IDs.

    Uses urllib.request for reliable streaming download (one persistent HTTP
    connection, 4 MB chunks, 120 s socket timeout per chunk) — avoids the
    fsspec per-range-request timeout that caused failures on large row groups.

    After download the file is filtered locally with pyarrow row-group selection:
    row groups whose _id min/max range provably excludes all target IDs are
    skipped, saving memory and decode time.

    The result (just the matching rows) is cached to .beir_cache/ so subsequent
    runs (e.g. after a Space restart) load instantly without re-downloading.
    """
    try:
        import pyarrow.parquet as pq
        import pyarrow.compute as pc
        import pyarrow as pa
    except ImportError as exc:
        log.warning(f"    pyarrow not available ({exc}), falling back to streaming")
        return _stream_fetch(en_hub, config, split, ids_set)

    parquet_url = _get_parquet_url(en_hub, config, split)
    if not parquet_url:
        log.warning(
            f"    No Parquet URL for {en_hub} [{config}/{split}], falling back to streaming"
        )
        return _stream_fetch(en_hub, config, split, ids_set)

    log.info(
        f"    Downloading {en_hub} [{config}/{split}] Parquet"
        f" ({len(ids_set)} IDs needed, one-time download then cached)..."
    )

    # --- Step 1: stream-download to a temp file ---
    import tempfile
    tmp_fd, tmp_path_str = tempfile.mkstemp(suffix=".parquet")
    tmp_path = Path(tmp_path_str)
    os.close(tmp_fd)

    try:
        req = urllib.request.Request(
            parquet_url, headers={"User-Agent": "load_nanobeir/1.0"}
        )
        downloaded = 0
        with urllib.request.urlopen(req, timeout=120) as resp:
            total = int(resp.headers.get("Content-Length", 0) or 0)
            size_str = f" / {total // 1024 // 1024} MB" if total else ""
            with open(tmp_path, "wb") as f_out:
                while True:
                    chunk = resp.read(4 * 1024 * 1024)  # 4 MB chunks
                    if not chunk:
                        break
                    f_out.write(chunk)
                    downloaded += len(chunk)
                    # Log every ~200 MB
                    if downloaded % (200 * 1024 * 1024) < 4 * 1024 * 1024:
                        pct = (
                            f" ({100 * downloaded // total}%)" if total else ""
                        )
                        log.info(
                            f"    ...{downloaded // 1024 // 1024} MB"
                            f"{size_str}{pct} downloaded"
                        )
        log.info(
            f"    Download complete ({downloaded // 1024 // 1024} MB)."
            f" Filtering for {len(ids_set)} IDs..."
        )

        # --- Step 2: smart row-group selection on the local file ---
        pf = pq.ParquetFile(tmp_path)
        meta = pf.metadata

        sample_id = next(iter(ids_set))
        is_numeric = sample_id.lstrip("-").isdigit()
        target_ints: set[int] = set()
        if is_numeric:
            try:
                target_ints = {int(i) for i in ids_set}
            except ValueError:
                is_numeric = False

        rgs_to_read: list[int] = []
        for rg_idx in range(meta.num_row_groups):
            row_grp = meta.row_group(rg_idx)
            include = True
            for col_idx in range(row_grp.num_columns):
                col = row_grp.column(col_idx)
                if col.path_in_schema != "_id" or not col.statistics:
                    continue
                rg_min = str(col.statistics.min)
                rg_max = str(col.statistics.max)
                if is_numeric:
                    try:
                        rg_min_int, rg_max_int = int(rg_min), int(rg_max)
                        if rg_min_int <= rg_max_int:
                            include = any(
                                rg_min_int <= t <= rg_max_int for t in target_ints
                            )
                        else:
                            # integer min > max → string-sorted file, use string cmp
                            include = any(rg_min <= t <= rg_max for t in ids_set)
                    except ValueError:
                        include = True
                else:
                    include = any(rg_min <= t <= rg_max for t in ids_set)
                break
            if include:
                rgs_to_read.append(rg_idx)

        pct = len(rgs_to_read) * 100 // meta.num_row_groups if meta.num_row_groups else 100
        log.info(
            f"    Row-group filter: {len(rgs_to_read)}/{meta.num_row_groups}"
            f" groups ({pct}%) contain target IDs."
        )

        id_array = pa.array(sorted(ids_set))
        tables: list = []
        for rg_idx in rgs_to_read:
            batch = pf.read_row_group(rg_idx, columns=["_id", "title", "text"])
            mask = pc.is_in(batch["_id"], value_set=id_array)
            filtered = batch.filter(mask)
            if len(filtered) > 0:
                tables.append(filtered)

    finally:
        try:
            tmp_path.unlink()
        except OSError:
            pass

    result: dict = {}
    if tables:
        table = pa.concat_tables(tables)
        result = {
            row["_id"]: {
                "title": row.get("title", "") or "",
                "text": row.get("text", "") or "",
            }
            for row in table.to_pylist()
        }

    found = len(result)
    missing = len(ids_set) - found
    msg = f"    Found {found}/{len(ids_set)} EN texts"
    if missing:
        msg += f" — {missing} IDs not found in BeIR source"
    log.info(msg)
    return result


def stream_en_texts(en_hub: str, config: str, split: str, ids_needed: set) -> dict:
    """
    Fetch EN texts for ids_needed from a BeIR dataset.

    First call: streams the dataset (potentially millions of rows for large corpora),
    caches ALL found records to .beir_cache/ as a small JSON file.

    Subsequent calls: loads from the cache file instantly — critical since the
    Argilla Space resets periodically and this script must be re-run without
    re-downloading gigabytes of data each time.

    Large corpora that benefit most from caching:
      BeIR/msmarco       8.8M rows
      BeIR/fever         5.4M rows
      BeIR/climate-fever 5.4M rows
      BeIR/hotpotqa      5.2M rows
      BeIR/dbpedia-entity 4.6M rows
      BeIR/nq            2.7M rows

    Returns dict: str(_id) -> {'title': str, 'text': str}
    """
    ids_set = {str(i) for i in ids_needed}
    cache_path = _cache_path(en_hub, config, split)

    # --- Cache hit ---
    if cache_path.exists():
        log.info(f"    Loading {len(ids_set)} IDs from cache: {cache_path.name}")
        cached = _load_cache(cache_path)
        result = {k: v for k, v in cached.items() if k in ids_set}
        found = len(result)
        missing = len(ids_set) - found
        msg = f"    Cache hit: {found}/{len(ids_set)} EN texts"
        if missing:
            msg += f" ({missing} IDs not in cache — may not exist in BeIR source)"
        log.info(msg)
        return result

    # --- Cache miss: fetch and populate ---
    if en_hub in API_FETCH_CORPORA and config == "corpus":
        cached_all = _smart_parquet_fetch(en_hub, config, split, ids_set)
    else:
        cached_all = _stream_fetch(en_hub, config, split, ids_set)

    _save_cache(cache_path, cached_all)
    log.info(f"    Cached to {cache_path}")

    return cached_all


def load_positives_by_query(bench: dict) -> dict:
    """
    Load relevance data and return {query_id: [corpus_id, ...]} mapping.

    Two formats depending on the source:
    - 'relevance' config (individual NanoX-bm25 datasets):
        one row per query with 'positive-corpus-ids' as a list
    - 'qrels' config (NanoBEIR-sr for NanoClimateFEVER):
        one row per (query, corpus) pair, flat
    """
    sr_hub = bench["sr_hub"]
    sr_split = bench["sr_split"]
    rel_config = bench["rel_config"]

    positives: dict = defaultdict(list)

    if rel_config == "qrels":
        ds = load_dataset(sr_hub, "qrels", split=sr_split)
        for row in ds:
            positives[str(row["query-id"])].append(str(row["corpus-id"]))
    else:
        ds = load_dataset(sr_hub, "relevance", split=sr_split)
        for row in ds:
            positives[str(row["query-id"])] = [
                str(i) for i in row["positive-corpus-ids"]
            ]

    return dict(positives)


# ---------------------------------------------------------------------------
# Per-benchmark record collector
# ---------------------------------------------------------------------------
def collect_benchmark_records(bench: dict, max_pos: int) -> list:
    """
    Fetch all data for one NanoBEIR benchmark and return its Argilla records.

    Does NOT upload — the caller collects records from all benchmarks,
    shuffles them, and uploads once to the merged dataset.

    Record IDs are prefixed with the benchmark name (e.g. NanoArguAna_query_42)
    so records from different benchmarks never collide in the merged dataset,
    and --only reruns can safely upsert without touching other benchmarks' records.
    """
    name = bench["name"]
    sr_hub = bench["sr_hub"]
    sr_split = bench["sr_split"]
    en_hub = bench["en_hub"]

    log.info(f"\n{'=' * 60}")
    log.info(f"  {name}")
    log.info(f"{'=' * 60}")

    # ------------------------------------------------------------------
    # 1. Serbian queries
    # ------------------------------------------------------------------
    log.info("  [1/6] Loading SR queries...")
    sr_q_ds = load_dataset(sr_hub, "queries", split=sr_split)
    sr_q_by_id = {str(r["_id"]): r["text"] for r in sr_q_ds}
    log.info(f"    {len(sr_q_by_id)} queries.")

    # ------------------------------------------------------------------
    # 2. English queries (streaming)
    # ------------------------------------------------------------------
    log.info("  [2/6] Fetching EN queries...")
    en_q_by_id = stream_en_texts(
        en_hub, "queries", bench["en_queries_split"], set(sr_q_by_id.keys())
    )

    # ------------------------------------------------------------------
    # 3. Relevance data → positive passage IDs
    # ------------------------------------------------------------------
    log.info("  [3/6] Loading relevance data...")
    positives_by_query = load_positives_by_query(bench)
    total_pos = sum(len(v) for v in positives_by_query.values())
    log.info(f"    {total_pos} total positive pairs across {len(positives_by_query)} queries.")

    # ------------------------------------------------------------------
    # 4. Cap positives per query and collect unique passage IDs
    # ------------------------------------------------------------------
    needed_passage_ids: set = set()
    for pos_ids in positives_by_query.values():
        needed_passage_ids.update(pos_ids[:max_pos])
    log.info(
        f"  [4/6] Capped at {max_pos}/query -> {len(needed_passage_ids)} unique passage IDs to annotate."
    )

    # ------------------------------------------------------------------
    # 5. Serbian corpus (only the passages we need)
    # ------------------------------------------------------------------
    log.info("  [5/6] Loading SR corpus (needed passages only)...")
    sr_corpus_ds = load_dataset(sr_hub, "corpus", split=sr_split)
    sr_c_by_id = {
        str(r["_id"]): r["text"]
        for r in sr_corpus_ds
        if str(r["_id"]) in needed_passage_ids
    }
    log.info(f"    {len(sr_c_by_id)} SR passages loaded.")

    # ------------------------------------------------------------------
    # 6. English corpus (streaming — may be millions of rows)
    # ------------------------------------------------------------------
    log.info("  [6/6] Fetching EN corpus (streaming)...")
    en_c_by_id = stream_en_texts(
        en_hub, "corpus", bench["en_corpus_split"], needed_passage_ids
    )

    # ------------------------------------------------------------------
    # Build Argilla records
    # ------------------------------------------------------------------
    records = []
    skipped_q = 0
    skipped_p = 0

    # Query records
    # IDs are benchmark-prefixed so records from different benchmarks never
    # collide in the merged dataset (e.g. NanoArguAna_query_42).
    for q_id, sr_text in sr_q_by_id.items():
        en = en_q_by_id.get(q_id, {})
        en_text = format_source(en.get("title", ""), en.get("text", ""))
        if not en_text or not sr_text:
            skipped_q += 1
            continue
        records.append(
            rg.Record(
                id=f"{name}_query_{q_id}",
                fields={
                    "annotation_guide_link": GUIDE_LINK_FIELD,
                    "source_text_en": en_text,
                    "translated_text_sr": sr_text,
                },
                suggestions=[
                    rg.Suggestion(
                        question_name="corrected_text_sr",
                        value=sr_text,
                        agent="DeepSeek-V3",
                        type="model",
                    )
                ],
                metadata={
                    "task_id": f"query_{q_id}",
                    "record_type": "query",
                    "benchmark": name,
                },
            )
        )

    # Passage records
    for p_id in needed_passage_ids:
        sr_text = sr_c_by_id.get(p_id)
        en = en_c_by_id.get(p_id, {})
        en_text = format_source(en.get("title", ""), en.get("text", ""))
        if not en_text or not sr_text:
            skipped_p += 1
            continue
        records.append(
            rg.Record(
                id=f"{name}_passage_{p_id}",
                fields={
                    "annotation_guide_link": GUIDE_LINK_FIELD,
                    "source_text_en": en_text,
                    "translated_text_sr": sr_text,
                },
                suggestions=[
                    rg.Suggestion(
                        question_name="corrected_text_sr",
                        value=sr_text,
                        agent="DeepSeek-V3",
                        type="model",
                    )
                ],
                metadata={
                    "task_id": f"passage_{p_id}",
                    "record_type": "passage",
                    "benchmark": name,
                },
            )
        )

    skipped_total = skipped_q + skipped_p
    log.info(
        f"  Built {len(records)} records "
        f"({len(sr_q_by_id) - skipped_q} queries + "
        f"{len(needed_passage_ids) - skipped_p} passages"
        + (f", {skipped_total} skipped — EN text not found" if skipped_total else "")
        + ")."
    )
    return records


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Load all 13 NanoBEIR benchmarks into a single merged Argilla dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--api-url", default=os.getenv("ARGILLA_API_URL"), help="Argilla API URL"
    )
    parser.add_argument(
        "--api-key", default=os.getenv("ARGILLA_API_KEY"), help="Argilla API key"
    )
    parser.add_argument("--workspace", default="argilla", help="Argilla workspace")
    parser.add_argument(
        "--dataset-name",
        default="NanoBEIR-sr",
        help="Name of the merged Argilla dataset to create or update",
    )
    parser.add_argument(
        "--max-pos",
        type=int,
        default=10,
        help="Max positive passages per query (caps annotation scope for high-positive datasets)",
    )
    parser.add_argument(
        "--only",
        default=None,
        metavar="BENCHMARK",
        help=(
            "Collect and upload records from only this benchmark "
            "(e.g. NanoArguAna). Upserts into the existing merged dataset — "
            "useful for adding a benchmark that previously failed."
        ),
    )
    parser.add_argument(
        "--no-shuffle",
        action="store_true",
        help="Upload records in collection order (benchmark by benchmark) instead of shuffled.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print record counts per benchmark without uploading anything.",
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help=(
            "Delete the existing dataset before creating a fresh one. "
            "All existing records and annotations will be permanently lost."
        ),
    )
    args = parser.parse_args()

    if not args.dry_run and (not args.api_url or not args.api_key):
        log.error("--api-url and --api-key are required (or set ARGILLA_API_URL / ARGILLA_API_KEY)")
        sys.exit(1)

    # Resolve benchmark list
    benchmarks_to_run = BENCHMARKS
    if args.only:
        benchmarks_to_run = [b for b in BENCHMARKS if b["name"] == args.only]
        if not benchmarks_to_run:
            valid = [b["name"] for b in BENCHMARKS]
            log.error(f"Unknown benchmark '{args.only}'. Valid names: {valid}")
            sys.exit(1)

    # Connect
    if not args.dry_run:
        client = rg.Argilla(api_url=args.api_url, api_key=args.api_key)
        log.info(f"Connected to Argilla at {args.api_url}")
    else:
        client = None
        log.info("[DRY RUN] Not connecting to Argilla.")

    log.info(
        f"Running {len(benchmarks_to_run)} benchmark(s) | "
        f"max_pos={args.max_pos} | dataset={args.dataset_name} | workspace={args.workspace}"
    )

    # Collect records from all benchmarks
    all_records = []
    results = {}
    failed = []

    for bench in benchmarks_to_run:
        try:
            records = collect_benchmark_records(bench=bench, max_pos=args.max_pos)
            all_records.extend(records)
            results[bench["name"]] = len(records)
        except Exception as exc:
            import traceback
            log.error(f"FAILED {bench['name']}: {exc}")
            traceback.print_exc()
            failed.append(bench["name"])

    # Summary
    log.info(f"\n{'=' * 60}")
    log.info("SUMMARY")
    log.info(f"{'=' * 60}")
    total = 0
    for name, n in results.items():
        log.info(f"  {name:<25} {n:>5} records")
        total += n
    log.info(f"  {'TOTAL':<25} {total:>5} records")
    if failed:
        log.error(f"  FAILED: {failed}")
        log.error(f"  Rerun with: --only <benchmark_name>")
    log.info(f"{'=' * 60}")

    if args.dry_run:
        return

    # Shuffle so annotators see a random mix from all benchmarks
    if not args.no_shuffle:
        import random
        random.shuffle(all_records)
        log.info(f"Shuffled {len(all_records)} records.")

    # Delete existing dataset if --recreate was requested
    if args.recreate:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            existing = client.datasets(name=args.dataset_name, workspace=args.workspace)
        if existing is not None:
            log.info(f"--recreate: deleting existing dataset '{args.dataset_name}'...")
            existing.delete()
            log.info("  Deleted.")
        else:
            log.info(f"--recreate: dataset '{args.dataset_name}' does not exist, nothing to delete.")

    # Upload all records to the single merged dataset
    dataset = get_or_create_dataset(client, args.dataset_name, args.workspace)
    log.info(f"Uploading {len(all_records)} records to '{args.dataset_name}'...")
    dataset.records.log(all_records)
    log.info("Done.")


if __name__ == "__main__":
    main()
