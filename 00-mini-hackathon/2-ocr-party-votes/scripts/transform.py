import json
import csv
import pandas as pd

def thai_to_arabic(thai_num):
    thai_digits = '๐๑๒๓๔๕๖๗๘๙'
    arabic_digits = '0123456789'
    translation_table = str.maketrans(thai_digits, arabic_digits)
    return thai_num.translate(translation_table)

def clean_json_data(docs_data):
    for doc_id, rows in docs_data.items():
        for item in rows:
            if "no" in item and isinstance(item["no"], str):
                item["no"] = thai_to_arabic(item["no"])
            if "score_num" in item and isinstance(item["score_num"], str):
                item["score_num"] = thai_to_arabic(item["score_num"])
    return docs_data

def from_json_to_csv(json_file, csv_file):
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['doc_id', 'no', 'name', 'score_num', 'score_text']) 
        
        for doc_id, rows in data.items():
            for item in rows:
                try:
                    raw_no = str(item.get('no', '0'))
                    raw_score = str(item.get('score_num', '0'))
                    name = item.get('name', '')
                    score_text = item.get('score_text', '')

                    clean_no = int(raw_no.replace(',', '').strip())
                    clean_score = int(raw_score.replace(',', '').strip())

                    writer.writerow([doc_id, clean_no, name, clean_score, score_text])
                except ValueError:
                    continue

def merge_to_submission(extracted_csv, submission_csv, output_csv):
    df_extracted = pd.read_csv(extracted_csv)
    df_sub = pd.read_csv(submission_csv)

    df_extracted['id'] = df_extracted['doc_id'] + '_' + df_extracted['no'].astype(str)
    
    mapping = df_extracted.set_index('id')['score_num'].to_dict()
    df_sub['votes'] = df_sub['id'].map(mapping).fillna(df_sub['votes']).astype(int)

    df_sub.to_csv(output_csv, index=False)