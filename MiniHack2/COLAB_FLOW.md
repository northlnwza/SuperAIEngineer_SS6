# Colab Flow

Copy each section below into a separate Google Colab cell.

This flow uses the same project logic as the local pipeline:

1. install dependencies
2. mount Google Drive
3. enter the project folder
4. set the Typhoon OCR API key
5. run a smoke test
6. run the full pipeline
7. generate the Kaggle-ready submission
8. inspect validation and optional safe-patched output

Replace the Drive path and API key before running.

## Cell 1: Title And Method

```python
# SuperAI Engineer OCR 2569
# Team: YOUR_TEAM_NAME
#
# Method summary:
# - Use Typhoon OCR to extract markdown tables from election form PNGs
# - Parse rows into structured party vote records
# - Align rows to the official submission template
# - Validate against page-1 total votes
# - Export a Kaggle-ready id,votes CSV
```

## Cell 2: Mount Google Drive

```python
from google.colab import drive
drive.mount("/content/drive")
```

## Cell 3: Go To Project Folder

```python
PROJECT_DIR = "/content/drive/MyDrive/MiniHack2"
%cd $PROJECT_DIR
```

## Cell 4: Install Dependencies

```python
!pip install -r requirements.txt
```

## Cell 5: Set API Key

```python
import os

# Replace this with your real key before running.
os.environ["TYPHOON_OCR_API_KEY"] = "YOUR_TYHOON_OCR_API_KEY"

print("api key set:", bool(os.environ.get("TYPHOON_OCR_API_KEY")))
```

## Cell 6: Verify Files And Environment

```python
!python scripts/check_api_key.py
!python scripts/inspect_docs.py
```

## Cell 7: Smoke Test On One Document

```python
!python scripts/build_submission.py --doc-id party_list_10_2
```

## Cell 8: Inspect Smoke Test Outputs

```python
!python scripts/summarize_validation.py --show 20
```

```python
!head -n 20 outputs/submissions/submission_typhoon_baseline.csv
```

```python
!head -n 20 outputs/submissions/submission_kaggle.csv
```

## Cell 9: Run The Full Pipeline

```python
!python scripts/build_submission.py
```

## Cell 10: Summarize Full Validation

```python
!python scripts/summarize_validation.py --show 100
```

## Cell 11: Rank The Best Docs To Inspect Manually

```python
!python scripts/rank_patch_queue.py --show 25
```

```python
!python scripts/rank_patch_queue.py --status needs_review_total_mismatch --show 20
```

```python
!python scripts/rank_patch_queue.py --status needs_review_missing_rows --show 20
```

## Cell 12: Inspect Suspicious Rows In Top Docs

```python
!python scripts/inspect_patch_docs.py --top-from-rank 10 --show-rows 8
```

## Cell 13: Suggest Low-Risk Patches

```python
!python scripts/suggest_patches.py --top 20 --max-changes 3
```

## Cell 14: Create A Conservative Second Submission

```python
!python scripts/apply_safe_patches.py
```

## Cell 15: Preview Final Kaggle Files

```python
!head -n 20 outputs/submissions/submission_kaggle.csv
```

```python
!head -n 20 outputs/submissions/submission_kaggle_safe_patched.csv
```

## Cell 16: Show Exact Output Paths

```python
from pathlib import Path

paths = [
    Path("outputs/submissions/submission_kaggle.csv"),
    Path("outputs/submissions/submission_kaggle_safe_patched.csv"),
    Path("outputs/debug/validation_report.json"),
    Path("outputs/debug/parsed_rows.json"),
]

for path in paths:
    print(path.resolve(), "exists=", path.exists())
```

## Cell 17: Optional Download From Colab

```python
from google.colab import files

# Uncomment one of these if you want to download directly from Colab.
# files.download("outputs/submissions/submission_kaggle.csv")
# files.download("outputs/submissions/submission_kaggle_safe_patched.csv")
```

## What To Submit

For Kaggle:

- upload `outputs/submissions/submission_kaggle.csv`
- or try `outputs/submissions/submission_kaggle_safe_patched.csv` as another candidate

For the competition form:

- share the Google Colab link as `Viewer`
- download the notebook as `.ipynb`

## Notes

- `submission_typhoon_baseline.csv` is the debug-friendly internal CSV with helper columns.
- `submission_kaggle.csv` is the real Kaggle-ready file with only `id,votes`.
- The OCR cache is stored in `outputs/ocr_raw/`, so reruns are cheaper after the first pass.
