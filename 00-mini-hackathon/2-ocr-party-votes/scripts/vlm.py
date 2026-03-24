import google.generativeai as genai
import json as std_json
import time
import os

def process_with_vlm(documents, api_key, model_name="gemini-1.5-flash", max_retries=3, retry_delay=5):
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)
    
    response_schema = {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "no": {"type": "string"},
                "name": {"type": "string"},
                "score_num": {"type": "string"},
                "score_text": {"type": "string"}
            },
            "required": ["no", "name", "score_num", "score_text"]
        }
    }

    system_prompt = """คุณคือระบบ AI อัจฉริยะสำหรับสกัดข้อมูลจากฟอร์มราชการ (ส.ส. ๕/๑)
ฉันจะส่งภาพเอกสารที่มาเป็นชุดเดียวกันให้คุณ (อาจมีตั้งแต่ 1 ถึง 7 หน้า)
กฎเหล็ก:
1. เพิกเฉยต่อ 'หน้าปก' หรือ 'หน้าว่าง' ที่ไม่มีตารางคะแนน
2. ดึงข้อมูลเฉพาะ "ตารางคะแนนผู้สมัคร/พรรคการเมือง" ถ้ารูปมีตารางต่อเนื่องหลายหน้า ให้ดึงมาต่อกันเป็นก้อนเดียว
3. รูปแบบข้อมูล: หมายเลข (no), ชื่อ (name), ตัวเลขคะแนน (score_num), คำอ่านคะแนน (score_text)
4. ตอบกลับเป็น JSON ตาม Schema เท่านั้น"""

    results_db = {}
    os.makedirs("results", exist_ok=True)

    print(f"\nStarting VLM Extraction ({model_name})...")
    
    for doc in documents:
        doc_id = doc['doc_id']
        paths = doc['pages']
        checkpoint_file = f"results/{doc_id}.json"

        if os.path.exists(checkpoint_file):
            print(f"Skipped [ {doc_id} ] (มี Checkpoint แล้ว)")
            with open(checkpoint_file, "r", encoding="utf-8") as f:
                results_db[doc_id] = std_json.load(f)
            continue

        print(f"กำลังสแกน [ {doc_id} ] ({doc['num_pages']} หน้า)...", end=" ", flush=True)
        success = False
        
        for attempt in range(max_retries):
            uploaded_files = []
            try:
                uploaded_files = [genai.upload_file(path=p) for p in paths]
                response = model.generate_content(
                    [system_prompt] + uploaded_files,
                    generation_config=genai.GenerationConfig(
                        response_mime_type="application/json",
                        response_schema=response_schema,
                        temperature=0.0
                    )
                )
                
                extracted_data = std_json.loads(response.text)
                results_db[doc_id] = extracted_data
                
                with open(checkpoint_file, "w", encoding="utf-8") as f:
                    std_json.dump(extracted_data, f, ensure_ascii=False, indent=4)
                
                print(f"สำเร็จ! ({len(extracted_data)} แถว)")
                success = True
                break
                
            except Exception as e:
                print(f"\n   Error (รอบ {attempt + 1}/{max_retries}): {e}")
                time.sleep(retry_delay)
            finally:
                for f in uploaded_files:
                    try: genai.delete_file(f.name)
                    except: pass

        if not success:
            print(f"ข้าม [ {doc_id} ] (ล้มเหลวหลังลอง {max_retries} ครั้ง)")
            
        time.sleep(4)
        
    return results_db