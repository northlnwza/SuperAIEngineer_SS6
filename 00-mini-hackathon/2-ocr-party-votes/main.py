from dotenv import load_dotenv
import logging
import json
import os

from scripts.file import ( 
    list_png_files, filter_valid_files, map_parse, 
    group_by_doc_id, sort_pages, to_documents 
)
from scripts.vlm import process_with_vlm
from scripts.transform import clean_json_data, from_json_to_csv, merge_to_submission

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

def build_pipeline(image_dir):
    files = list_png_files(image_dir)
    valid_files = filter_valid_files(files)
    parsed = map_parse(valid_files)
    grouped = group_by_doc_id(parsed)
    sorted_docs = sort_pages(grouped)
    return to_documents(sorted_docs)

def main():
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    
    if not api_key:
        logger.error("GEMINI_API_KEY is missing from environment variables.")
        raise ValueError("Missing required API key.")
    
    logger.info("Initializing Election Data Extraction Pipeline")

    logger.info("Phase 1/4: Preparing image documents")
    all_docs = build_pipeline("./sample")
    logger.info(f"Identified {len(all_docs)} target documents for processing")

    logger.info("Phase 2/4: Executing VLM extraction process")
    raw_data = process_with_vlm(all_docs, api_key, model_name="gemini-1.5-flash")

    logger.info("Phase 3/4: Transforming data and standardizing formats")
    clean_data = clean_json_data(raw_data)
    
    json_filename = "all_election_arabic.json"
    csv_filename = "all_election_arabic.csv"
    
    with open(json_filename, "w", encoding="utf-8") as f:
        json.dump(clean_data, f, ensure_ascii=False, indent=4)
        
    from_json_to_csv(json_filename, csv_filename)
    logger.info(f"Aggregated data saved to {csv_filename}")

    logger.info("Phase 4/4: Merging results into submission template")
    merge_to_submission(
        extracted_csv=csv_filename,
        submission_csv="submission.csv",
        output_csv="final_submission_ready.csv"
    )

    logger.info("Pipeline execution completed successfully. Ready for submission.")

if __name__ == "__main__":    
    main()