"""
Setup script for the Serbian translation annotation dataset in Argilla v2.

Matches the actual annotation guidelines:
- Fields: English source, Serbian MT translation
- Questions: quality rating (1-5), corrected translation, comment
- Guidelines: full Serbian annotation instructions

Usage:
    python setup_dataset.py --api-url <URL> --api-key <KEY>
"""

import argparse
import os
import sys

import argilla as rg


GUIDELINES = """
# Uputstvo za ispravljanje automatskog prevoda (MS MARCO)

## Opis podataka

Skup podataka je MS MARCO v1.1, jedan od najkorišćenijih benchmark-ova za pretragu informacija i odgovaranje na pitanja. Originalni skup sadrži pitanja iz Bing pretrage sa odgovorima zasnovanim na web pasusima. Automatski prevod sa engleskog na srpski urađen je korišćenjem DeepSeek-V3.

Vaš zadatak je da proverite i ispravite kvalitet automatskog prevoda upita (queries) i pasusa (passages) iz ovog skupa podataka.

## Pravila

- Ispravljajte svaki primer zasebno i nezavisno od drugih primera.
- Ispravku primera radite samostalno bez korišćenja LLM-ova (ChatGPT, Gemini, Claude, i sl.).
- Ne koristite LLM-ove za proveru ispravki teksta koje ste uneli.
- Za proveru tačnosti prevoda imenovanih entiteta i stručne terminologije koristite Google pretragu.

## Proces anotacije

### Step 1: Procena ispravnosti teksta (Engleski original naspram srpskog prevoda)

Prvo pročitajte tekst na engleskom, zatim proverite prevod na srpskom. Proverite da li:
- Tekst na srpskom ima isto značenje kao tekst na engleskom
- Informacije iz teksta na engleskom nisu izostavljene u srpskom
- U srpskom prevodu ne postoji dodat tekst ili informacije koje ne postoje u originalnom engleskom tekstu
- Namera, ton, i semantičko značenje originalnog teksta na engleskom je isto i u prevedenom srpskom tekstu

### Step 2: Kvalitet jezika (Samo srpski prevod)

Ponovo pročitajte samo tekst na srpskom, i proverite:
- Da li je prevod u skladu sa pravilima srpske gramatike, sintakse, i leksike
- Da li je rod, broj, i padež imenskih reči tačan
- Da li su glagoli u ispravnom rodu, broju i vremenu
- Da li su imenovani entiteti (lična imena, geografske lokacije, nazivi institucija, itd.) prevedeni tačno
- Da li su koreference u tekstu tačno razrešene
- Da li u prevodu postoje pravopisne greške

### Step 3: Unos ispravki, ocene kvaliteta prevoda i komentara

**Ispravke:** Unesite ispravljeni prevod u polje za ispravke. Ako prevodu nisu potrebne ispravke, unesite tačno: No corrections.

**Komentar:** Napišite kratak i jasan komentar u kojem objašnjavate unete ispravke (npr. ispravka gramatičke greške, dodat izostavljeni deo teksta, ispravljeno pogrešno značenje, ispravak terminologije, itd.). Ako nema ispravki: No corrections needed.

**Ocena kvaliteta (1-5):**

1 - Potpuno netačan prevod: Prevod ne prenosi značenje originalnog teksta. Većina informacija u prevodu ne postoji u tekstu na engleskom.

2 - Prevod sadrži veće greške: Prevod menja značenje originalnog teksta. Segmenti teksta su pogrešno prevedeni, nedostaju, ili je terminologija ključna za razumevanje pogrešno prevedena.

3 - Adekvatan prevod sa manjim greškama: Prevedeni tekst tačno prenosi značenje originalnog teksta, ali sadrži greške koje mogu uticati na razumevanje teksta.

4 - Zadovoljavajuć kvalitet: Prevod potpuno i tačno prenosi značenje. Sadrži manje stilske ili sintaktičke greške koje ne utiču na razumevanje. Imenovani entiteti i terminologija su većinom tačno prevedeni.

5 - Odličan kvalitet: Prevod u potpunosti prenosi značenje teksta. Prevod je prirodan na srpskom jeziku, imenovani entiteti i terminologija su potpuno tačno prevedeni. Nema gramatičkih, stilskih, ili sintaksičkih grešaka.

### Skraćeni opis ocena

- **Low (1-2):** Prevod je netačan ili sadrži veće greške. Značenje je promenjeno, ključne informacije nedostaju ili je terminologija pogrešno prevedena.
- **Medium (3):** Prevod je adekvatan, ali sa primetnim nedostacima. Značenje je tačno preneto, ali postoje greške koje mogu otežati razumevanje.
- **High (4-5):** Prevod je visokog kvaliteta. Značenje je potpuno i tačno preneto, terminologija i imenovani entiteti su precizni. Tekst zvuči prirodno na srpskom jeziku.
"""


def create_dataset(api_url: str, api_key: str, workspace: str = "argilla",
                   dataset_name: str = "translation-annotation-sr"):
    """Create the translation annotation dataset in Argilla."""

    client = rg.Argilla(api_url=api_url, api_key=api_key)

    # Check if dataset already exists
    try:
        existing = client.datasets(name=dataset_name, workspace=workspace)
        if existing is not None:
            print(f"Dataset '{dataset_name}' already exists in workspace '{workspace}'.")
            print("Delete it first if you want to recreate, or use a different name.")
            return existing
    except Exception:
        pass

    settings = rg.Settings(
        guidelines=GUIDELINES,
        fields=[
            rg.TextField(
                name="source_text_en",
                title="English Source Text (Originalni tekst na engleskom)",
                use_markdown=False,
                required=True,
            ),
            rg.TextField(
                name="translated_text_sr",
                title="Machine Translation (Mašinski prevod na srpski)",
                use_markdown=False,
                required=True,
            ),
        ],
        questions=[
            rg.RatingQuestion(
                name="quality_score",
                title="Ocena kvaliteta prevoda",
                description=(
                    "1 = Potpuno netačan prevod, "
                    "2 = Veće greške, "
                    "3 = Adekvatan sa manjim greškama, "
                    "4 = Zadovoljavajuć kvalitet, "
                    "5 = Odličan kvalitet"
                ),
                values=[1, 2, 3, 4, 5],
                required=True,
            ),
            rg.TextQuestion(
                name="corrected_text_sr",
                title="Ispravite prevod sa engleskog na srpski",
                description=(
                    'Unesite ispravljeni prevod. Ako prevod ne zahteva ispravke, '
                    'unesite: No corrections.'
                ),
                required=True,
                use_markdown=False,
            ),
            rg.TextQuestion(
                name="comment",
                title="Komentar",
                description=(
                    "Kratko objašnjenje ispravki (npr. ispravka gramatičke greške, "
                    "dodat izostavljeni deo teksta, ispravljeno pogrešno značenje, "
                    "ispravak terminologije, itd.). "
                    'Ako nema ispravki, unesite: No corrections needed.'
                ),
                required=True,
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
                name="source_dataset",
                title="Source Dataset",
                visible_for_annotators=False,
            ),
        ],
        allow_extra_metadata=True,
    )

    dataset = rg.Dataset(
        name=dataset_name,
        workspace=workspace,
        settings=settings,
        client=client,
    )

    dataset.create()
    print(f"Dataset '{dataset_name}' created in workspace '{workspace}'.")
    return dataset


def main():
    parser = argparse.ArgumentParser(description="Set up Argilla translation annotation dataset")
    parser.add_argument("--api-url", default=os.getenv("ARGILLA_API_URL"))
    parser.add_argument("--api-key", default=os.getenv("ARGILLA_API_KEY"))
    parser.add_argument("--workspace", default="argilla")
    parser.add_argument("--dataset-name", default="translation-annotation-sr-v2")
    args = parser.parse_args()

    if not args.api_url or not args.api_key:
        print("Error: --api-url and --api-key required (or set env vars)")
        sys.exit(1)

    create_dataset(args.api_url, args.api_key, args.workspace, args.dataset_name)


if __name__ == "__main__":
    main()