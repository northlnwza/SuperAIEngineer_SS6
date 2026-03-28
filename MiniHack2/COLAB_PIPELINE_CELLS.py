# %% [markdown]
# # SuperAI Engineer OCR 2569
# Actual Colab pipeline code using the same project logic as the working local solution.
#
# Copy each `# %%` block into a separate Google Colab cell.

# %% [markdown]
# ## Cell 1: Mount Google Drive

from google.colab import drive

drive.mount("/content/drive")


# %% [markdown]
# ## Cell 2: Enter Project Directory
# Update `PROJECT_DIR` to match your Google Drive path.

PROJECT_DIR = "/content/drive/MyDrive/MiniHack2"

%cd $PROJECT_DIR


# %% [markdown]
# ## Cell 3: Install Dependencies

!pip install -r requirements.txt


# %% [markdown]
# ## Cell 4: Set API Key
# Replace the value with your real Typhoon OCR key.

import os

os.environ["TYPHOON_OCR_API_KEY"] = "YOUR_TYPHOON_OCR_API_KEY"
print("api key set:", bool(os.environ.get("TYPHOON_OCR_API_KEY")))


# %% [markdown]
# ## Cell 5: Import The Actual Pipeline Modules

from pathlib import Path
import csv
from IPython.display import Markdown, display
import json
import sys

ROOT = Path(PROJECT_DIR)
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from minihack2.documents import group_document_pages
from minihack2.ocr_client import ocr_image_to_markdown
from minihack2.parse_tables import choose_best_row, parse_rows
from minihack2.pipeline import build_submission
from minihack2.validate import extract_expected_total_votes


# %% [markdown]
# ## Cell 6: Show The Project Modules Used In The Pipeline
# This prints the exact source files used by the notebook pipeline.

module_files = [
    ("documents.py", ROOT / "src" / "minihack2" / "documents.py"),
    ("ocr_client.py", ROOT / "src" / "minihack2" / "ocr_client.py"),
    ("parse_tables.py", ROOT / "src" / "minihack2" / "parse_tables.py"),
    ("normalize.py", ROOT / "src" / "minihack2" / "normalize.py"),
    ("thai_number_words.py", ROOT / "src" / "minihack2" / "thai_number_words.py"),
    ("validate.py", ROOT / "src" / "minihack2" / "validate.py"),
    ("pipeline.py", ROOT / "src" / "minihack2" / "pipeline.py"),
]

for label, path in module_files:
    display(Markdown(f"### {label}"))
    print(path)
    print(path.read_text(encoding="utf-8"))


# %% [markdown]
# ## Cell 7: Define Standard Paths

images_dir = ROOT / "data" / "images"
template_path = ROOT / "data" / "submission_template.csv"
cache_dir = ROOT / "outputs" / "ocr_raw"
debug_json_path = ROOT / "outputs" / "debug" / "parsed_rows.json"
validation_json_path = ROOT / "outputs" / "debug" / "validation_report.json"
baseline_output_csv = ROOT / "outputs" / "submissions" / "submission_typhoon_baseline.csv"
kaggle_output_csv = ROOT / "outputs" / "submissions" / "submission_kaggle.csv"

print(images_dir)
print(template_path)


# %% [markdown]
# ## Cell 8: Inspect How Input Pages Are Grouped Into Documents

grouped_pages = group_document_pages(images_dir)
print("document_count =", len(grouped_pages))

sample_doc_id = next(iter(grouped_pages))
print("sample_doc_id =", sample_doc_id)
print("sample_pages =", [page.path.name for page in grouped_pages[sample_doc_id]])


# %% [markdown]
# ## Cell 9: OCR One Page And See Raw Markdown
# This shows what Typhoon OCR returns before parsing.

sample_page = grouped_pages["party_list_10_2"][0].path
sample_markdown = ocr_image_to_markdown(sample_page)

print("page =", sample_page.name)
print(sample_markdown[:3000])


# %% [markdown]
# ## Cell 10: Parse OCR Rows From One Page
# This is the actual table parsing step.

sample_rows = parse_rows(sample_markdown, source=sample_page.name)
print("parsed_row_count =", len(sample_rows))
for row in sample_rows[:5]:
    print(
        {
            "row_num": row.row_num,
            "party_name": row.party_name,
            "votes": row.votes,
            "vote_words_value": row.vote_words_value,
            "source": row.source,
        }
    )


# %% [markdown]
# ## Cell 11: Inspect Page-1 Total Extraction
# This is the page-level validation anchor used later.

page1_path = grouped_pages["party_list_10_2"][0].path
page1_markdown = ocr_image_to_markdown(page1_path)
expected_total_votes = extract_expected_total_votes(page1_markdown)

print("doc_id =", "party_list_10_2")
print("expected_total_votes =", expected_total_votes)


# %% [markdown]
# ## Cell 12: Show How Alignment Works For One Template Row
# This demonstrates row matching between OCR output and the official template.

with template_path.open("r", encoding="utf-8-sig", newline="") as f:
    template_rows = list(csv.DictReader(f))

target_rows = [row for row in template_rows if row["doc_id"] == "party_list_10_2"]
target_row = target_rows[0]

best_match = choose_best_row(
    sample_rows,
    expected_row_num=int(target_row["row_num"]),
    expected_party_name=target_row["party_name"],
    prefer_party_name=False,
)

print("template_row =", target_row)
print(
    "best_match =",
    {
        "row_num": best_match.row_num if best_match else None,
        "party_name": best_match.party_name if best_match else None,
        "votes": best_match.votes if best_match else None,
        "vote_words_value": best_match.vote_words_value if best_match else None,
        "source": best_match.source if best_match else None,
    },
)


# %% [markdown]
# ## Cell 13: Run A Smoke-Test Submission Build
# This uses the full pipeline logic exactly as the project does.

build_submission(
    images_dir=images_dir,
    template_path=template_path,
    output_csv=baseline_output_csv,
    cache_dir=cache_dir,
    debug_json_path=debug_json_path,
    validation_json_path=validation_json_path,
    doc_ids={"party_list_10_2"},
    max_docs=None,
)

print("baseline_output_csv exists =", baseline_output_csv.exists())
print("debug_json_path exists =", debug_json_path.exists())
print("validation_json_path exists =", validation_json_path.exists())


# %% [markdown]
# ## Cell 14: Inspect Debug JSON For The Smoke Test

debug_data = json.loads(debug_json_path.read_text(encoding="utf-8"))
doc_debug = debug_data["party_list_10_2"]

print("pages =", doc_debug["pages"])
print("expected_total_votes =", doc_debug["expected_total_votes"])
print("validation =", doc_debug["validation"])
print("first_assigned_rows =")
for row in doc_debug["assigned_rows"][:5]:
    print(row)


# %% [markdown]
# ## Cell 15: Run The Full Dataset
# This is the actual full pipeline run used to create the final result.

build_submission(
    images_dir=images_dir,
    template_path=template_path,
    output_csv=baseline_output_csv,
    cache_dir=cache_dir,
    debug_json_path=debug_json_path,
    validation_json_path=validation_json_path,
    doc_ids=None,
    max_docs=None,
)

print("full baseline built:", baseline_output_csv)


# %% [markdown]
# ## Cell 16: Convert Internal Baseline CSV To Kaggle Format
# Kaggle only accepts `id,votes`.

with baseline_output_csv.open("r", encoding="utf-8-sig", newline="") as f:
    internal_rows = list(csv.DictReader(f))

kaggle_output_csv.parent.mkdir(parents=True, exist_ok=True)
with kaggle_output_csv.open("w", encoding="utf-8-sig", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["id", "votes"])
    writer.writeheader()
    for row in internal_rows:
        writer.writerow({"id": row["id"], "votes": row["votes"]})

print("kaggle_output_csv =", kaggle_output_csv)


# %% [markdown]
# ## Cell 17: Inspect Final Validation Summary

validation_data = json.loads(validation_json_path.read_text(encoding="utf-8"))

status_counts = {}
for item in validation_data.values():
    status_counts[item["status"]] = status_counts.get(item["status"], 0) + 1

print("status_counts =", status_counts)

flagged = [item for item in validation_data.values() if item["status"] != "ok"]
flagged.sort(key=lambda item: (item["status"], item["doc_id"]))

for item in flagged[:20]:
    print(
        item["doc_id"],
        item["status"],
        f"matched={item['matched_rows']}/{item['expected_rows']}",
        f"total={item['extracted_total_votes']}/{item['expected_total_votes']}",
        f"missing_rows={item['missing_row_nums']}",
    )


# %% [markdown]
# ## Cell 18: Preview The Final Kaggle Submission

with kaggle_output_csv.open("r", encoding="utf-8-sig", newline="") as f:
    reader = csv.reader(f)
    for idx, row in enumerate(reader):
        print(row)
        if idx >= 9:
            break


# %% [markdown]
# ## Cell 19: Optional Download

from google.colab import files

# Uncomment to download the final Kaggle-ready file.
# files.download(str(kaggle_output_csv))
