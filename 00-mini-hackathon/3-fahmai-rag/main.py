"""
Ultra High-Accuracy RAG System for FahMai QA
Best Practices:
1. Two-stage verification (retrieve + verify each choice)
2. Multiple models with self-consistency
3. Chain-of-Thought with explicit reasoning
4. Choice-by-choice validation against KB
5. Numeric value extraction and cross-checking
"""

from scripts.requests import ask_llm, parse_answer 
from scripts.rag import get_knowledge_base
from dotenv import load_dotenv
import pandas as pd
import os
import re
import logging
from datetime import datetime
from collections import Counter

load_dotenv()

# ============================================================
# LOGGING SETUP
# ============================================================

def setup_logging():
    """Setup logging to file and console."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"run_{timestamp}.log"
    
    # Create logger
    logger = logging.getLogger("fahmai")
    logger.setLevel(logging.DEBUG)
    
    # File handler - detailed logs
    fh = logging.FileHandler(log_filename, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s | %(message)s"))
    
    # Console handler - brief output
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(message)s"))
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger, log_filename

LOG, LOG_FILE = setup_logging()

# ============================================================
# SYSTEM PROMPTS
# ============================================================

ANALYSIS_PROMPT = """คุณเป็นผู้เชี่ยวชาญด้านสินค้าอิเล็กทรอนิกส์ร้านฟ้าใหม่ (FahMai)

## กฎการตัดสินใจ

### ขั้นที่ 1: จำแนกคำถาม
- คำถามเกี่ยวกับ "ร้านฟ้าใหม่" (สินค้า, นโยบาย, บริการ) → ดำเนินต่อ
- คำถามไม่เกี่ยวกับร้านฟ้าใหม่ (เช่น วันหยุดราชการ, ตั๋วเครื่องบิน, สูตรอาหาร, ดอกเบี้ย) → ตอบ 10 ทันที

### ขั้นที่ 2: ค้นหาข้อมูลในฐานความรู้
- หาข้อมูลที่ตอบคำถามได้จากฐานความรู้
- จดค่าที่ถูกต้อง (ตัวเลข, สเปค, ราคา, นโยบาย)

### ขั้นที่ 3: ตรวจสอบตัวเลือก 1-8 ทีละข้อ
ตัวเลือกจะถูกต้องก็ต่อเมื่อ:
- ข้อมูลทุกจุดตรงกับฐานความรู้ 100%
- ไม่มีตัวเลข/สเปคที่ผิด
- ไม่มีข้อมูลที่แต่งขึ้นมา

ตัวเลือกที่ผิด:
- มีตัวเลขผิดแม้เพียงนิดเดียว (เช่น 10 ATM vs 5 ATM, 3,990 vs 2,990)
- IP rating ไม่ถูกต้อง (IP68 vs IP69K)
- ชื่อรุ่นผิด (X9 vs X9 Pro)
- มีข้อมูลถูกบางส่วนแต่ผิดบางส่วน → ถือว่าผิดทั้งข้อ

### ขั้นที่ 4: สรุปคำตอบ
- มีตัวเลือกที่ถูกต้อง 100% → เลือกตัวเลือกนั้น
- ไม่มีตัวเลือกใดถูกต้อง แต่คำถามเกี่ยวกับฟ้าใหม่ → ตอบ 9
- คำถามไม่เกี่ยวกับฟ้าใหม่ → ตอบ 10

## กับดักที่พบบ่อย
- ตัวเลือกใช้ตัวเลขที่คล้ายแต่ผิด
- ตัวเลือกอ้างชื่อรุ่นที่คล้ายกันแต่ผิดรุ่น
- ตัวเลือกมีข้อมูลถูกครึ่งเดียว
- ตัวเลือกอ้างนโยบายที่ดูสมจริงแต่ไม่มีในฐานความรู้

## รูปแบบคำตอบ
<analysis>
1. คำถามถามเรื่อง: [สรุปประเด็น]
2. ข้อมูลจากฐานความรู้: [ข้อมูลที่เกี่ยวข้อง พร้อมค่าที่ถูกต้อง]
3. ตรวจสอบตัวเลือก:
   - ตัวเลือก 1: [ถูก/ผิด เพราะ...]
   - ตัวเลือก 2: [ถูก/ผิด เพราะ...]
   ...
4. สรุป: [เหตุผลที่เลือกคำตอบ]
</analysis>
<answer>[1-10]</answer>"""


VERIFY_PROMPT = """คุณเป็นผู้ตรวจสอบคำตอบร้านฟ้าใหม่ กรุณาตรวจสอบว่าคำตอบที่เสนอถูกต้องหรือไม่

## ฐานความรู้
{context}

## คำถาม
{question}

## ตัวเลือก
{choices}

## คำตอบที่เสนอ: {proposed_answer}

## คำสั่ง
1. ตรวจสอบว่าตัวเลือกที่ {proposed_answer} ตรงกับข้อมูลในฐานความรู้หรือไม่
2. ถ้าถูกต้อง ยืนยันคำตอบเดิม
3. ถ้าพบว่าผิด ระบุคำตอบที่ถูกต้อง

<verification>
[ตรวจสอบความถูกต้องของตัวเลือก {proposed_answer}]
</verification>
<answer>[1-10]</answer>"""


# ============================================================
# HELPER FUNCTIONS
# ============================================================

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
    return "\n".join([f"{i}. {c}" for i, c in enumerate(choices, 1)])


def build_prompt(question: str, choices: list, context: str) -> str:
    """Build the full prompt."""
    return f"""=== ฐานความรู้ร้านฟ้าใหม่ ===
{context}

=== คำถาม ===
{question}

=== ตัวเลือก ===
{format_choices(choices)}

กรุณาวิเคราะห์อย่างละเอียดและตอบ:"""


def extract_answer(response: str) -> int:
    """Extract answer from response."""
    if response is None:
        return None
    
    # Try to find <answer> tag
    match = re.search(r'<answer>\s*(\d+)\s*</answer>', response, re.IGNORECASE)
    if match:
        ans = int(match.group(1))
        if 1 <= ans <= 10:
            return ans
    
    # Fallback: find last standalone number 1-10
    nums = re.findall(r'\b(\d+)\b', response)
    for n in reversed(nums):
        if 1 <= int(n) <= 10:
            return int(n)
    
    return None


def is_unrelated_question(question: str, choices: list) -> bool:
    """Check if question is clearly unrelated to FahMai."""
    unrelated_keywords = [
        'วันหยุดราชการ', 'ตั๋วเครื่องบิน', 'ดอกเบี้ยเงินฝาก', 'สูตร', 
        'ผัดกระเพรา', 'ข้าวผัด', 'ทำอาหาร', 'ซักผ้า', 'สูตรอาหาร',
        'โควิด', 'covid', 'วันชาติ', 'ปีใหม่', 'สงกรานต์',
        'วันแม่', 'วันพ่อ', 'วันเด็ก', 'วันหยุด',
        'บัตรเครดิตธนาคาร', 'สินเชื่อบ้าน', 'ดอกเบี้ย'
    ]
    q_lower = question.lower()
    
    # Check question text
    for kw in unrelated_keywords:
        if kw in q_lower or kw in question:
            return True
    
    return False


def majority_vote(answers: list) -> int:
    """Simple majority vote."""
    if not answers:
        return 9
    
    valid = [a for a in answers if a is not None]
    if not valid:
        return 9
    
    counter = Counter(valid)
    return counter.most_common(1)[0][0]


def weighted_vote(answers: list, weights: list) -> int:
    """Weighted majority vote."""
    if not answers:
        return 9
    
    valid_pairs = [(a, w) for a, w in zip(answers, weights) if a is not None]
    if not valid_pairs:
        return 9
    
    counter = Counter()
    for ans, weight in valid_pairs:
        counter[ans] += weight
    
    return counter.most_common(1)[0][0]


# ============================================================
# MAIN FUNCTION
# ============================================================

def process_question(idx: int, question: str, choices: list, context: str, api_key: str) -> int:
    """Process a single question with multiple verification rounds."""
    
    LOG.debug(f"Q{idx+1} Context docs: {len(context)} chars")
    
    # Quick check for unrelated questions
    if is_unrelated_question(question, choices):
        LOG.info(f"  → Detected unrelated question → 10")
        LOG.debug(f"Q{idx+1} UNRELATED → 10")
        return 10
    
    user_prompt = build_prompt(question, choices, context)
    
    # Stage 1: Get answers from multiple models
    thaillm_models = ["typhoon", "openthaigpt", "kbtg"]
    answers = []
    weights = []
    model_responses = {}
    
    # ThaiLLM models
    for model in thaillm_models:
        print(f"  {model}...", end=" ", flush=True)
        response = ask_llm(
            api_key=api_key,
            model=model,
            messages=[
                {"role": "system", "content": ANALYSIS_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
        )
        answer = extract_answer(response)
        print(f"→ {answer}")
        
        model_responses[model] = {"answer": answer, "response": response[:500] if response else None}
        LOG.debug(f"Q{idx+1} {model} → {answer}")
        
        if answer is not None:
            answers.append(answer)
            weights.append(1.0)
    
    # Check for consensus
    if len(set(answers)) == 1 and len(answers) >= 2:
        # All agree - high confidence
        final = answers[0]
        LOG.info(f"  Consensus: {final}")
        LOG.debug(f"Q{idx+1} CONSENSUS → {final}")
        return final
    
    # Stage 2: Disagreement - do verification round with typhoon
    if answers:
        proposed = majority_vote(answers)
        LOG.info(f"  Disagreement, verifying {proposed}...")
        print(f"  Disagreement, verifying {proposed}...", end=" ", flush=True)
        
        verify_prompt = VERIFY_PROMPT.format(
            context=context,
            question=question,
            choices=format_choices(choices),
            proposed_answer=proposed
        )
        
        response = ask_llm(
            api_key=api_key,
            model="typhoon",
            messages=[
                {"role": "user", "content": verify_prompt}
            ],
        )
        
        verify_answer = extract_answer(response)
        print(f"→ {verify_answer}")
        LOG.debug(f"Q{idx+1} verify1 → {verify_answer}")
        
        if verify_answer is not None:
            answers.append(verify_answer)
            weights.append(1.5)  # Higher weight for verification
    
    # Stage 3: If still uncertain, do second verification with different model
    unique_answers = set(a for a in answers if a is not None)
    if len(unique_answers) > 1:
        print(f"  Still uncertain, second verify...", end=" ", flush=True)
        response = ask_llm(
            api_key=api_key,
            model="openthaigpt",
            messages=[
                {"role": "system", "content": ANALYSIS_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
        )
        
        answer = extract_answer(response)
        print(f"→ {answer}")
        LOG.debug(f"Q{idx+1} verify2 → {answer}")
        
        if answer is not None:
            answers.append(answer)
            weights.append(1.2)
    
    # Final weighted vote
    final = weighted_vote(answers, weights)
    LOG.debug(f"Q{idx+1} FINAL → {final} (votes: {answers}, weights: {weights})")
    return final


def main():
    api_key = os.getenv("API_KEY")
    
    
    LOG.info("Loading knowledge base...")
    kb = get_knowledge_base("data/knowledge_base")
    
    questions_df = pd.read_csv("data/questions.csv")
    submission = []
    
    for idx, row in questions_df.iterrows():
        question = row["question"]
        choices = extract_choices(row)
        
        LOG.info(f"\n{'=' * 70}")
        LOG.info(f"Q{idx+1}: {question[:70]}...")
        LOG.debug(f"Q{idx+1} FULL: {question}")
        
        # Get comprehensive context
        context = kb.get_relevant_context(question, choices, max_chars=18000)
        
        # Process with verification
        final_answer = process_question(idx, question, choices, context, api_key)
        
        submission.append((idx + 1, final_answer))
        LOG.info(f"  FINAL: {final_answer}")
    
    # Save results
    submission_df = pd.DataFrame(submission, columns=["id", "answer"])
    submission_df.to_csv("submission.csv", index=False)
    
    LOG.info("\n" + "=" * 70)
    LOG.info("Created submission.csv successfully!")
    LOG.info(f"Log saved to: {LOG_FILE}")
    
    # Show answer distribution
    answer_counts = Counter([s[1] for s in submission])
    LOG.info("\nAnswer distribution:")
    for ans in sorted(answer_counts.keys()):
        LOG.info(f"  {ans}: {answer_counts[ans]} questions")
    
    # Log summary to file
    LOG.debug("=" * 70)
    LOG.debug("FINAL ANSWERS SUMMARY")
    LOG.debug("=" * 70)
    for qid, ans in submission:
        LOG.debug(f"Q{qid}: {ans}")


if __name__ == "__main__":
    main()
