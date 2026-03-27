from scripts.requests import ask_llm, parse_answer 
from scripts.voiting import majority_vote 
from scripts.rag import get_knowledge_base
from dotenv import load_dotenv
import pandas as pd
import os

load_dotenv()

SYSTEM_PROMPT = """คุณเป็นผู้เชี่ยวชาญตรวจสอบข้อมูลสินค้าร้านฟ้าใหม่ (FahMai)
ทำหน้าที่: เลือกคำตอบที่ถูกต้องจากตัวเลือก 1-10 โดยอิงจากข้อมูลฐานความรู้เท่านั้น

## กฎเหล็ก (ห้ามละเมิด)
1. ตอบเป็นตัวเลข 1-10 เท่านั้น ไม่มีคำอธิบาย
2. ใช้เฉพาะข้อมูลจากฐานความรู้ที่ให้มา ห้ามใช้ความรู้ภายนอก
3. ตัวเลือกต้องถูกต้องทุกรายละเอียด ผิดแม้แต่ 1 จุดก็ถือว่าผิดทั้งข้อ
4. ข้อ 9 = "ไม่มีข้อมูลในฐานข้อมูล" → ใช้เมื่อคำถามเกี่ยวกับฟ้าใหม่แต่ไม่พบข้อมูล
5. ข้อ 10 = "คำถามไม่เกี่ยวกับฟ้าใหม่" → ใช้เมื่อคำถามไม่เกี่ยวกับร้าน/สินค้า/บริการฟ้าใหม่เลย

## กับดักที่ต้องระวัง
- ตัวเลือกอาจผสมข้อมูลถูกและผิดในข้อเดียวกัน → ต้องตรวจทุกจุด
- ตัวเลข/ราคา/สเปคอาจผิดเพียงนิดเดียว (เช่น 10 ATM vs 5 ATM)
- ชื่อรุ่นที่คล้ายกัน (เช่น X9 vs X9 Pro vs X9 Pro Max)
- IP rating ที่ต่างกัน (IP68 vs IP69K)
- ข้อมูลประกัน/นโยบายที่อาจเขียนให้ดูสมจริงแต่ไม่ตรงกับข้อมูลจริง

## วิธีตัดสินใจ
1. อ่านคำถาม → หาประเด็นหลักที่ต้องการคำตอบ
2. ค้นหาข้อมูลในฐานความรู้ → จดค่าที่ถูกต้อง
3. ตรวจสอบทุกตัวเลือก 1-8 เทียบกับข้อมูลจริง
4. เลือกตัวเลือกที่ข้อมูลตรง 100%
5. ถ้าไม่มีตัวเลือกใดตรง → ตอบ 9
6. ถ้าคำถามไม่เกี่ยวกับฟ้าใหม่เลย → ตอบ 10"""


def extract_choices(row) -> list:
    """Extract choices 1-10 from dataframe row."""
    choices = []
    for i in range(1, 11):
        col_name = f"choice_{i}"
        if col_name in row:
            choices.append(str(row[col_name]))
    return choices


def format_choices(choices: list) -> str:
    """Format choices for display."""
    formatted = []
    for i, choice in enumerate(choices, 1):
        formatted.append(f"{i}. {choice}")
    return "\n".join(formatted)


def build_prompt(question: str, choices: list, context: str) -> str:
    """Build the full prompt with question, choices, and context."""
    return f"""=== ข้อมูลฐานความรู้ร้านฟ้าใหม่ (ใช้อ้างอิงเท่านั้น) ===
{context}

=== คำถาม ===
{question}

=== ตัวเลือก (เลือก 1 ข้อที่ถูกต้อง 100%) ===
{format_choices(choices)}

คำตอบ (ตัวเลข 1-10 เท่านั้น):"""


def main():
    api_key = os.getenv("API_KEY")
    models = ["typhoon", "openthaigpt", "kbtg"]
    
    print("Loading knowledge base...")
    kb = get_knowledge_base("data/knowledge_base")
    
    question_list = pd.read_csv("data/questions.csv")
    submission = []

    for index, row in question_list.iterrows():
        question = row["question"]
        choices = extract_choices(row)
        
        print(f"\n{'='*60}")
        print(f"Q{index + 1}: {question[:100]}...")
        
        context = kb.get_relevant_context(question, choices)
        user_prompt = build_prompt(question, choices, context)
        
        answers = []
        for model in models:
            print(f"  {model}...", end=" ")
            response = ask_llm(
                api_key=api_key,
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
            )
            answer = parse_answer(response)
            print(f"→ {answer}")
            answers.append(answer)

        valid_answers = [a for a in answers if a is not None]
        if valid_answers:
            final_answer = majority_vote(valid_answers)
        else:
            final_answer = 9
            
        submission.append((index + 1, final_answer))
        print(f"  FINAL: {final_answer} (votes: {answers})")
    
    submission_df = pd.DataFrame(submission, columns=["id", "answer"])
    submission_df.to_csv("submission.csv", index=False)

    print("\n" + "="*60)
    print("Created submission.csv successfully!")


if __name__ == "__main__":
    main()