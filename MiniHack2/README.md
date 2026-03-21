# SuperAI Engineer OCR 2569 Starter

This workspace contains a Typhoon OCR based pipeline for extracting vote counts from Thai election forms and turning them into a Kaggle submission.

The core flow is:

1. group PNG pages into logical `doc_id` documents
2. OCR each page with Typhoon OCR and cache the markdown
3. parse table rows from the OCR output
4. align those rows to `submission_template.csv`
5. validate against page-1 totals and missing rows
6. write submission CSVs and debug artifacts

## Why this baseline is a good fit

Typhoon's current OCR docs say the recommended endpoint is `typhoon-ocr` and that `ocr_document(...)` returns structured, layout-aware Markdown. The release note also highlights Thai government-form and table handling, which is a good match for Form สส.6/1.

Sources:

- [Typhoon OCR docs](https://docs.opentyphoon.ai/en/ocr/)
- [Typhoon quickstart](https://docs.opentyphoon.ai/en/quickstart/)
- [Typhoon rate limits](https://docs.opentyphoon.ai/en/rate-limits/)
- [Typhoon OCR 1.5 release note](https://opentyphoon.ai/blog/th/typhoon-ocr-release)

## Setup

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Set your API key:

```bash
$env:TYPHOON_OCR_API_KEY="your_api_key_here"
```

3. Optional sanity check:

```bash
python scripts/check_api_key.py
```

## Core Pipeline

The core implementation lives in:

- `src/minihack2/documents.py`
- `src/minihack2/ocr_client.py`
- `src/minihack2/parse_tables.py`
- `src/minihack2/normalize.py`
- `src/minihack2/thai_number_words.py`
- `src/minihack2/validate.py`
- `src/minihack2/pipeline.py`

### What each module does

`documents.py`

- groups page images into one logical document by `doc_id`
- combines files like `party_list_10_2.png` and `party_list_10_2_page2.png`

`ocr_client.py`

- calls Typhoon OCR
- caches page OCR output as markdown in `outputs/ocr_raw/`

`parse_tables.py`

- parses HTML or markdown table output from Typhoon
- extracts structured rows like `row_num`, `party_name`, `votes`, `source`
- chooses the best OCR row for each expected template row

`normalize.py`

- converts Thai digits to Arabic digits
- strips punctuation and keeps numeric vote strings clean

`thai_number_words.py`

- parses Thai number words written in parentheses
- lets the pipeline compare digit OCR against written-value OCR

`validate.py`

- extracts the page-1 total vote count from section `4.1`
- compares sum of extracted rows against the expected total
- marks each document as `ok`, `needs_review_missing_rows`, `needs_review_total_mismatch`, or `warning_missing_page1_total`

`pipeline.py`

- orchestrates the whole run
- does OCR, parsing, alignment, rescue logic, validation, and file writing

## Main Run Commands

Full run:

```bash
python scripts/build_submission.py
```

Single document smoke test:

```bash
python scripts/build_submission.py --doc-id party_list_10_2
```

Small subset in template order:

```bash
python scripts/build_submission.py --max-docs 10
```

## Output Files

After a run, inspect these files first:

`outputs/ocr_raw/`

- one cached OCR markdown file per page
- best place to inspect what Typhoon actually returned

`outputs/debug/parsed_rows.json`

- full debug view per `doc_id`
- includes:
  - page list
  - expected total votes
  - parsed OCR rows
  - assigned rows used for submission
  - validation summary for that document

`outputs/debug/validation_report.json`

- compact QC report for every document
- best place to find which docs need manual review

`outputs/submissions/submission_typhoon_baseline.csv`

- internal full-width CSV with helper columns:
  - `id`
  - `doc_id`
  - `row_num`
  - `party_name`
  - `votes`
- useful for debugging, not for Kaggle upload

`outputs/submissions/submission_kaggle.csv`

- Kaggle-ready file
- contains only:
  - `id`
  - `votes`

`outputs/submissions/submission_kaggle_safe_patched.csv`

- second Kaggle-ready candidate with a few conservative automatic row fixes applied

## Scripts For Inspection And Manual Debugging

These are the scripts you will use most often when debugging manually.

### `scripts/build_submission.py`

Purpose:

- runs the full pipeline
- refreshes submission output and debug JSON

Useful commands:

```bash
python scripts/build_submission.py
python scripts/build_submission.py --max-docs 30
python scripts/build_submission.py --doc-id constituency_18_2
python scripts/build_submission.py --doc-id party_list_11_1 --doc-id party_list_13_7
```

Use this whenever parser or rescue logic changes.

### `scripts/inspect_docs.py`

Purpose:

- quickly inspect dataset layout and document/page grouping

Use this first if you want to understand the raw input set before running OCR.

### `scripts/summarize_validation.py`

Purpose:

- shows how many docs are `ok`
- shows the first flagged docs with row coverage and total comparison

Useful commands:

```bash
python scripts/summarize_validation.py --show 30
```

This is the quickest health check after any run.

### `scripts/rank_patch_queue.py`

Purpose:

- ranks flagged docs by likely ROI for manual patching
- prioritizes small-gap and nearly-complete docs

Useful commands:

```bash
python scripts/rank_patch_queue.py --show 25
python scripts/rank_patch_queue.py --status needs_review_total_mismatch --show 20
python scripts/rank_patch_queue.py --status needs_review_missing_rows --show 20
```

Use this when you want to decide which docs are worth manual effort first.

### `scripts/inspect_patch_docs.py`

Purpose:

- inspects suspicious rows inside one or more docs
- highlights:
  - unmatched rows
  - rows on total-row pages
  - digit OCR vs Thai number word disagreements
  - rescued rows

Useful commands:

```bash
python scripts/inspect_patch_docs.py --doc-id party_list_13_7 --show-rows 8
python scripts/inspect_patch_docs.py --top-from-rank 10 --show-rows 8
```

Use this when a doc is flagged and you want to see the likely bad rows immediately.

### `scripts/suggest_patches.py`

Purpose:

- proposes row substitutions using Thai number words
- tries to reduce the document total mismatch
- does not modify files

Useful commands:

```bash
python scripts/suggest_patches.py --top 15 --max-changes 3
python scripts/suggest_patches.py --doc-id party_list_18_1
```

Use this to see whether a flagged doc has an obvious low-effort fix.

### `scripts/apply_safe_patches.py`

Purpose:

- applies only conservative automatic fixes to a Kaggle CSV
- writes a second candidate submission

Useful command:

```bash
python scripts/apply_safe_patches.py
```

Use this when you want a low-risk `v2` submission without editing the main baseline file.

### `scripts/check_api_key.py`

Purpose:

- confirms that `TYPHOON_OCR_API_KEY` is available

Useful command:

```bash
python scripts/check_api_key.py
```

## Recommended Manual Debug Flow

When a submission looks weak, this is the fastest manual review loop:

1. Rebuild or rerun the relevant docs.

```bash
python scripts/build_submission.py --doc-id party_list_13_7 --doc-id constituency_18_2
```

2. Check overall QC.

```bash
python scripts/summarize_validation.py --show 30
```

3. Rank the best docs to inspect.

```bash
python scripts/rank_patch_queue.py --show 20
```

4. Inspect suspicious rows inside those docs.

```bash
python scripts/inspect_patch_docs.py --doc-id party_list_13_7 --show-rows 8
```

5. Ask for suggested patch candidates.

```bash
python scripts/suggest_patches.py --doc-id party_list_13_7
```

6. If the safe fixes look reasonable, create a second submission candidate.

```bash
python scripts/apply_safe_patches.py
```

## Recommended OCR Usage

Use Typhoon OCR as a page-to-markdown/table extractor, not as a one-shot full-task generator.

Why:

- the competition score depends only on the final vote string
- the template already gives the target party rows
- structured OCR plus deterministic cleanup is easier to debug

The baseline code uses the official helper directly:

```python
from typhoon_ocr import ocr_document
markdown = ocr_document(pdf_or_image_path="page.png")
```

If you later switch to direct prompting, keep the OCR output faithful to the page structure and do numeric cleanup in your own code.

## Notes On Scale

Typhoon OCR is rate-limited and this dataset has hundreds of page images. One full uncached pass takes a while, so page-level OCR caching in `outputs/ocr_raw/` is essential.

Because OCR is cached, later reruns are much cheaper if you only change parsing, rescue logic, or submission writing.
