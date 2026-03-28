import argparse
import csv
import json
import os
import random
import re
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
from openai import APIConnectionError, APIStatusError, OpenAI, RateLimitError
from pythainlp.tokenize import word_tokenize
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm


ALLOWED_THAILLM_MODELS = {
    "OpenThaiGPT-ThaiLLM-8B-instruct-v7.2",
    "Pathumma-ThaiLLM-qwen3-8b-think-3.0.0",
    "Typhoon-S-ThaiLLM-8B-Instruct",
    "THaLLE-0.2-ThaiLLM-8b-fa",
}
DEFAULT_MODEL_NAMES = ["OpenThaiGPT-ThaiLLM-8B-instruct-v7.2"]
DEFAULT_EMBED_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

SYSTEM_PROMPT = """คุณเป็นระบบตอบคำถามหลายตัวเลือกของร้านฟ้าใหม่
กติกาสำคัญ:
1. ใช้เฉพาะข้อมูลจากบริบทที่ให้มาเท่านั้น
2. เลือกคำตอบได้เพียงหนึ่งข้อ เป็นเลข 1-10
3. ข้อ 9 ใช้เมื่อคำถามเกี่ยวข้องกับร้านฟ้าใหม่ แต่ข้อมูลในบริบทไม่พอจะตอบอย่างมั่นใจ
4. ข้อ 10 ใช้เมื่อคำถามไม่เกี่ยวข้องกับร้านฟ้าใหม่ สินค้า แบรนด์ในเครือ นโยบาย บริการ โปรโมชัน หรือข้อมูลร้าน
5. ถ้าตัวเลือกใดมีรายละเอียดเกินจริง แม้บางส่วนจะคล้ายข้อมูลจริง ก็ห้ามเลือกตัวเลือกนั้น
6. ถ้าบริบทมีหลักฐานตรงตัว เช่น ราคา รุ่น สเปก ระยะเวลา จำนวน หรือเงื่อนไข ให้ยึดหลักฐานตรงตัวนั้น
7. ตอบเป็น JSON บรรทัดเดียวเท่านั้น ในรูปแบบ {"answer": <1-10>, "confidence": <0-1>, "reason": "<สั้นมาก>"}"""

STORE_KEYWORDS = {
    "ฟ้าใหม่",
    "fahmai",
    "สายฟ้า",
    "ดาวเหนือ",
    "คลื่นเสียง",
    "วงโคจร",
    "จุดเชื่อม",
    "zenbyte",
    "novatech",
    "pulsegear",
    "wongkhojon",
    "judchuam",
    "daonuea",
    "saifah",
    "kluensiang",
    "อาร์คเวฟ",
    "arcwave",
}

POLICY_KEYWORDS = {
    "คืน",
    "refund",
    "ยกเลิก",
    "จัดส่ง",
    "ส่งของ",
    "ชำระ",
    "ผ่อน",
    "รับประกัน",
    "เคลม",
    "สมาชิก",
    "คะแนน",
    "points",
    "trade-in",
    "เทิร์น",
    "crypto",
    "cryptocurrency",
    "bitcoin",
    "คริปโต",
    "สาขา",
    "faq",
    "บริการ",
}

NON_FAHMAI_HINTS = {
    "วันหยุดราชการ",
    "ผลบอล",
    "สภาพอากาศ",
    "ราคาทอง",
    "หวย",
    "นายก",
    "ดารา",
}

THAI_STOPWORDS = {
    "คือ",
    "และ",
    "กับ",
    "ของ",
    "ที่",
    "ได้",
    "ไหม",
    "มั้ย",
    "หรือ",
    "ครับ",
    "ค่ะ",
    "คะ",
    "หน่อย",
    "หน่อยครับ",
    "หน่อยค่ะ",
    "อะไร",
    "อย่างไร",
    "เท่าไหร่",
    "กี่",
    "รุ่น",
    "สินค้า",
    "ร้าน",
    "ใน",
    "จาก",
    "ให้",
    "มี",
    "เป็น",
    "อยู่",
    "ได้ไหม",
}


def normalize_text(text: str) -> str:
    text = text.lower()
    text = text.replace("฿", " ")
    text = text.replace("×", "x")
    text = re.sub(r"[,]", "", text)
    text = re.sub(r"[\u200b\xa0]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def simplify_text(text: str) -> str:
    text = normalize_text(text)
    text = re.sub(r"[/_\\|:()\-]+", " ", text)
    text = re.sub(r"[^\w\s\u0E00-\u0E7F\.]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def tokenize_search(text: str) -> List[str]:
    normalized = re.sub(r"[/_\\|:()\-]+", " ", text.lower())
    thai_tokens = word_tokenize(normalized, engine="newmm", keep_whitespace=False)
    extra_tokens = re.findall(r"[a-z0-9\.]+", normalized)
    merged = thai_tokens + extra_tokens
    tokens = []
    for token in merged:
        token = token.strip()
        if not token:
            continue
        if token in THAI_STOPWORDS:
            continue
        if len(token) == 1 and not token.isdigit():
            continue
        tokens.append(token)
    return tokens


def extract_numbers_and_units(text: str) -> Set[str]:
    pattern = (
        r"\d+(?:\.\d+)?\s*"
        r"(?:atm|mah|mm|cm|m|เมตร|นิ้ว|hz|ghz|mhz|gb|tb|w|kg|กก\.?|ชั่วโมง|วันทำการ|วัน|ปี|เดือน|บาท|รายการ|ซิม|เครื่อง|nits?)?"
    )
    return {
        re.sub(r"\s+", "", item.strip())
        for item in re.findall(pattern, normalize_text(text), flags=re.IGNORECASE)
        if item.strip()
    }


def choice_head_text(choice_text: str) -> str:
    head = re.split(r"\s+[—\-]\s+| \(| เพราะ| โดย| ซึ่ง", choice_text, maxsplit=1)[0].strip()
    return head or choice_text.strip()


def choice_variants(choice_text: str) -> List[str]:
    variants = [choice_text.strip()]
    head = choice_head_text(choice_text)
    if head != choice_text.strip():
        variants.append(head)

    without_paren = re.sub(r"\([^)]*\)", "", choice_text).strip()
    if without_paren and without_paren not in variants:
        variants.append(without_paren)

    deduped: List[str] = []
    seen: Set[str] = set()
    for item in variants:
        normalized = normalize_text(item)
        if normalized and normalized not in seen:
            seen.add(normalized)
            deduped.append(item)
    return deduped


def extract_title(text: str, fallback: str) -> str:
    for line in text.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return fallback


def split_markdown_sections(text: str) -> List[Tuple[str, str]]:
    sections: List[Tuple[str, str]] = []
    current_heading = "ภาพรวม"
    buffer: List[str] = []

    for line in text.splitlines():
        if re.match(r"^##+\s+", line):
            content = "\n".join(buffer).strip()
            if content:
                sections.append((current_heading, content))
            current_heading = re.sub(r"^##+\s+", "", line).strip()
            buffer = []
        else:
            buffer.append(line)

    content = "\n".join(buffer).strip()
    if content:
        sections.append((current_heading, content))
    return sections


def split_into_blocks(text: str) -> List[str]:
    raw_blocks = re.split(r"\n\s*\n", text.strip())
    return [block.strip() for block in raw_blocks if block.strip()]


def pack_blocks(blocks: Sequence[str], char_budget: int = 1100) -> List[str]:
    packed: List[str] = []
    current: List[str] = []
    current_len = 0

    for block in blocks:
        block_len = len(block)
        if block_len > char_budget:
            if current:
                packed.append("\n\n".join(current))
                current = []
                current_len = 0
            for start in range(0, block_len, char_budget):
                packed.append(block[start : start + char_budget])
            continue

        if current and current_len + block_len + 2 > char_budget:
            packed.append("\n\n".join(current))
            current = [block]
            current_len = block_len
        else:
            current.append(block)
            current_len += block_len + 2

    if current:
        packed.append("\n\n".join(current))
    return packed


def slug_to_alias(stem: str) -> str:
    stem = re.sub(r"^[a-z]{2,3}(?:-[a-z]{2})?-\d+_", "", stem.lower())
    parts = [part for part in stem.split("_") if part]
    if parts and parts[0] in {
        "daonuea",
        "saifah",
        "judchuam",
        "kluensiang",
        "wongkhojon",
        "arcwave",
        "novatech",
        "pulsegear",
        "zenbyte",
    }:
        parts = parts[1:]
    return " ".join(parts).strip()


def extract_aliases(title: str, path: str) -> Set[str]:
    aliases: Set[str] = set()
    aliases.add(simplify_text(title))
    aliases.add(simplify_text(slug_to_alias(Path(path).stem)))
    aliases.add(simplify_text(re.sub(r"\([^)]*\)", "", title).strip()))

    for bit in re.findall(r"\(([^)]{2,})\)", title):
        aliases.add(simplify_text(bit))

    compact_aliases: Set[str] = set()
    for alias in aliases:
        if not alias:
            continue
        compact_aliases.add(alias)
        words = alias.split()
        if len(words) >= 2:
            compact_aliases.add(" ".join(words[-3:]))
            compact_aliases.add(" ".join(words[-2:]))
    return {alias for alias in compact_aliases if len(alias) >= 3}


def load_documents(kb_dir: Path) -> List[Dict]:
    documents: List[Dict] = []
    for path in sorted(kb_dir.rglob("*.md")):
        rel_path = path.relative_to(kb_dir).as_posix()
        text = path.read_text(encoding="utf-8")
        title = extract_title(text, path.stem)
        aliases = extract_aliases(title, rel_path)
        documents.append(
            {
                "path": rel_path,
                "text": text,
                "title": title,
                "category": rel_path.split("/", 1)[0],
                "aliases": aliases,
                "alias_tokens": set(tokenize_search(" ".join(sorted(aliases)))),
            }
        )
    return documents


def build_chunks(documents: Sequence[Dict], char_budget: int = 1100) -> List[Dict]:
    chunks: List[Dict] = []
    for doc_id, doc in enumerate(documents):
        sections = split_markdown_sections(doc["text"])
        for section_heading, section_text in sections:
            blocks = split_into_blocks(section_text)
            for chunk_id, block_group in enumerate(pack_blocks(blocks, char_budget=char_budget)):
                chunk_text = (
                    f"ชื่อเอกสาร: {doc['title']}\n"
                    f"หมวดหมู่: {doc['category']}\n"
                    f"แหล่งข้อมูล: {doc['path']}\n"
                    f"หัวข้อย่อย: {section_heading}\n\n"
                    f"{block_group}"
                )
                title_tokens = set(tokenize_search(doc["title"]))
                path_tokens = set(tokenize_search(doc["path"]))
                chunks.append(
                    {
                        "doc_id": doc_id,
                        "chunk_id": f"{doc_id}-{chunk_id}",
                        "source": doc["path"],
                        "title": doc["title"],
                        "category": doc["category"],
                        "section": section_heading,
                        "text": chunk_text,
                        "aliases": doc["aliases"],
                        "title_tokens": title_tokens | path_tokens | doc["alias_tokens"] | set(tokenize_search(section_heading)),
                    }
                )
    return chunks


def load_questions(data_dir: Path) -> List[Dict]:
    rows: List[Dict] = []
    with (data_dir / "questions.csv").open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows.append(
                {
                    "id": int(row["id"]),
                    "question": row["question"],
                    "choices": {str(i): row[f"choice_{i}"] for i in range(1, 11)},
                }
            )
    return rows


def build_bm25(chunks: Sequence[Dict]) -> Tuple[BM25Okapi, List[List[str]]]:
    tokenized_chunks = [tokenize_search(chunk["text"]) for chunk in chunks]
    return BM25Okapi(tokenized_chunks), tokenized_chunks


def build_embeddings(
    chunks: Sequence[Dict],
    embed_model_name: str = DEFAULT_EMBED_MODEL,
    batch_size: int = 64,
) -> Tuple[SentenceTransformer, np.ndarray]:
    model = SentenceTransformer(embed_model_name)
    texts = [chunk["text"] for chunk in chunks]
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    return model, np.asarray(embeddings)


def category_hints(question: str) -> Set[str]:
    normalized = normalize_text(question)
    hints: Set[str] = set()
    if any(term in normalized for term in POLICY_KEYWORDS):
        hints.add("policies")
    if any(term in normalized for term in ["สาขา", "ร้าน", "faq", "trade-in", "คริปโต", "crypto", "ติดต่อ"]):
        hints.add("store_info")
    if any(term in normalized for term in ["รุ่น", "ราคา", "สเปค", "ในกล่อง", "รองรับ", "กันน้ำ", "ram", "ssd"]):
        hints.add("products")
    if not hints:
        hints.update({"products", "policies", "store_info"})
    return hints


def detect_alias_matches(text: str, documents: Sequence[Dict]) -> Set[int]:
    normalized = simplify_text(text)
    matches: Set[int] = set()
    for doc_id, doc in enumerate(documents):
        for alias in doc["aliases"]:
            if alias and len(alias) >= 4 and alias in normalized:
                matches.add(doc_id)
                break
    return matches


def infer_candidate_doc_ids(question: str, choices: Dict[str, str], documents: Sequence[Dict]) -> Set[int]:
    matched = detect_alias_matches(question, documents)
    question_tokens = set(tokenize_search(question))

    for doc_id, doc in enumerate(documents):
        if doc_id in matched:
            continue
        if len(question_tokens & doc["alias_tokens"]) >= 2:
            matched.add(doc_id)

    if not matched:
        choice_heads = " ".join(choice_head_text(choices[str(i)]) for i in range(1, 9))
        matched = detect_alias_matches(choice_heads, documents)

    if not matched:
        allowed_categories = category_hints(question)
        matched = {idx for idx, doc in enumerate(documents) if doc["category"] in allowed_categories}

    return matched


def question_store_relevance(question: str, choices: Dict[str, str], documents: Sequence[Dict]) -> Tuple[float, Set[int]]:
    normalized_question = normalize_text(question)
    score = 0.0

    if any(keyword in normalized_question for keyword in STORE_KEYWORDS):
        score += 2.2
    if any(keyword in normalized_question for keyword in POLICY_KEYWORDS):
        score += 1.2
    if any(keyword in normalized_question for keyword in NON_FAHMAI_HINTS):
        score -= 1.5

    alias_matches = infer_candidate_doc_ids(question, choices, documents)
    if alias_matches:
        score += min(len(alias_matches), 3) * 1.25

    return score, alias_matches


def dense_retrieve(
    query: str,
    embed_model: SentenceTransformer,
    chunk_embeddings: np.ndarray,
    allowed_indices: Optional[Sequence[int]] = None,
    top_k: int = 12,
) -> Tuple[np.ndarray, np.ndarray]:
    query_embedding = embed_model.encode([query], normalize_embeddings=True)
    scores = np.dot(chunk_embeddings, query_embedding.T).reshape(-1)
    if allowed_indices is not None:
        allowed_indices = np.asarray(list(allowed_indices), dtype=int)
        filtered_scores = scores[allowed_indices]
        top_local = np.argsort(filtered_scores)[::-1][:top_k]
        return allowed_indices[top_local], filtered_scores[top_local]
    top_indices = np.argsort(scores)[::-1][:top_k]
    return top_indices, scores[top_indices]


def bm25_retrieve(
    query: str,
    bm25: BM25Okapi,
    allowed_indices: Optional[Sequence[int]] = None,
    top_k: int = 12,
) -> Tuple[np.ndarray, np.ndarray]:
    tokens = tokenize_search(query)
    scores = np.asarray(bm25.get_scores(tokens))
    if allowed_indices is not None:
        allowed_indices = np.asarray(list(allowed_indices), dtype=int)
        filtered_scores = scores[allowed_indices]
        top_local = np.argsort(filtered_scores)[::-1][:top_k]
        return allowed_indices[top_local], filtered_scores[top_local]
    top_indices = np.argsort(scores)[::-1][:top_k]
    return top_indices, scores[top_indices]


def question_relevance_bonus(question: str, chunk: Dict, preferred_categories: Set[str]) -> float:
    question_tokens = set(tokenize_search(question))
    overlap = len(question_tokens & chunk["title_tokens"])
    bonus = min(overlap, 6) * 0.05

    normalized_question = simplify_text(question)
    if any(alias and len(alias) >= 4 and alias in normalized_question for alias in chunk["aliases"]):
        bonus += 0.28

    source_slug = simplify_text(Path(chunk["source"]).stem.replace("_", " "))
    if source_slug and source_slug in normalized_question:
        bonus += 0.16

    if chunk["category"] in preferred_categories:
        bonus += 0.08

    if chunk["category"] == "policies" and any(
        term in normalized_question
        for term in ["คืน", "ยกเลิก", "จัดส่ง", "ผ่อน", "ชำระ", "รับประกัน", "สมาชิก", "คะแนน"]
    ):
        bonus += 0.06

    if chunk["category"] == "store_info" and any(
        term in normalized_question
        for term in ["สาขา", "ร้าน", "faq", "trade-in", "คริปโต", "crypto", "บริการ"]
    ):
        bonus += 0.06

    return bonus


def hybrid_retrieve(
    question: str,
    choices: Dict[str, str],
    documents: Sequence[Dict],
    chunks: Sequence[Dict],
    embed_model: SentenceTransformer,
    chunk_embeddings: np.ndarray,
    bm25: BM25Okapi,
    candidate_doc_ids: Optional[Set[int]] = None,
    final_k: int = 8,
    fetch_k: int = 20,
    rrf_k: int = 60,
) -> List[Dict]:
    del documents
    preferred_categories = category_hints(question)
    allowed_indices: Optional[List[int]] = None
    if candidate_doc_ids:
        allowed_indices = [idx for idx, chunk in enumerate(chunks) if chunk["doc_id"] in candidate_doc_ids]

    dense_idx, _ = dense_retrieve(
        question,
        embed_model,
        chunk_embeddings,
        allowed_indices=allowed_indices,
        top_k=fetch_k,
    )
    bm25_idx, _ = bm25_retrieve(
        question,
        bm25,
        allowed_indices=allowed_indices,
        top_k=fetch_k,
    )

    scores: Dict[int, float] = {}
    for rank, idx in enumerate(dense_idx, start=1):
        scores[int(idx)] = scores.get(int(idx), 0.0) + 1.0 / (rrf_k + rank)
    for rank, idx in enumerate(bm25_idx, start=1):
        scores[int(idx)] = scores.get(int(idx), 0.0) + 1.0 / (rrf_k + rank)

    choice_text = " ".join(choice_head_text(choices[str(i)]) for i in range(1, 9))
    choice_tokens = set(tokenize_search(choice_text))
    for idx in list(scores.keys()):
        scores[idx] += question_relevance_bonus(question, chunks[idx], preferred_categories)
        scores[idx] += min(len(choice_tokens & chunks[idx]["title_tokens"]), 4) * 0.03

    top_indices = sorted(scores, key=scores.get, reverse=True)[:final_k]
    return [chunks[idx] for idx in top_indices]


def build_context(chunks: Sequence[Dict]) -> str:
    parts = []
    for rank, chunk in enumerate(chunks, start=1):
        parts.append(
            f"[หลักฐาน {rank}]\n"
            f"แหล่งข้อมูล: {chunk['source']}\n"
            f"ชื่อเอกสาร: {chunk['title']}\n"
            f"หัวข้อย่อย: {chunk['section']}\n"
            f"{chunk['text']}"
        )
    return "\n\n".join(parts)


def score_choice_against_chunks(choice_text: str, retrieved_chunks: Sequence[Dict]) -> float:
    best_score = 0.0
    for chunk in retrieved_chunks:
        context = normalize_text(chunk["text"])
        context_tokens = set(tokenize_search(chunk["text"]))
        for variant in choice_variants(choice_text):
            normalized_choice = normalize_text(variant)
            score = 0.0

            if normalized_choice and normalized_choice in context:
                score += 5.0

            numbers = extract_numbers_and_units(variant)
            if numbers:
                matched_numbers = sum(1 for number in numbers if number in context.replace(" ", ""))
                score += matched_numbers * 1.25
                if matched_numbers == len(numbers):
                    score += 0.75
                elif matched_numbers == 0:
                    score -= 0.8

            informative_tokens = [
                token
                for token in tokenize_search(variant)
                if token not in STORE_KEYWORDS and token not in THAI_STOPWORDS and len(token) > 1
            ]
            matched = sum(1 for token in informative_tokens if token in context_tokens or token in context)
            if informative_tokens:
                coverage = matched / len(informative_tokens)
                score += coverage * 2.2
                if matched == len(informative_tokens) and len(informative_tokens) >= 2:
                    score += 0.7

            best_score = max(best_score, score)
    return best_score


def exact_choice_match(question: str, choices: Dict[str, str], retrieved_chunks: Sequence[Dict]) -> Optional[int]:
    del question
    support: List[Tuple[int, float]] = []
    for choice_id in range(1, 9):
        support.append((choice_id, score_choice_against_chunks(choices[str(choice_id)], retrieved_chunks)))
    support.sort(key=lambda item: item[1], reverse=True)
    best_choice, best_score = support[0]
    second_score = support[1][1] if len(support) > 1 else 0.0
    if best_score >= 5.5 and best_score - second_score >= 1.2:
        return best_choice
    return None


def parse_json_answer(raw_text: str) -> Optional[Dict]:
    if not raw_text:
        return None

    clean = re.sub(r"<think>.*?</think>", "", raw_text, flags=re.DOTALL).strip()
    match = re.search(r"\{.*\}", clean, flags=re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(0))
            answer = int(data.get("answer"))
            if 1 <= answer <= 10:
                confidence = data.get("confidence")
                try:
                    confidence = float(confidence) if confidence is not None else None
                except (TypeError, ValueError):
                    confidence = None
                return {
                    "answer": answer,
                    "confidence": confidence,
                    "reason": str(data.get("reason", "")).strip(),
                    "raw": clean,
                }
        except (json.JSONDecodeError, TypeError, ValueError):
            pass

    match = re.search(r"\b([1-9]|10)\b", clean)
    if match:
        answer = int(match.group(1))
        return {"answer": answer, "confidence": None, "reason": "", "raw": clean}
    return None


def build_prompt(
    question: str,
    choices: Dict[str, str],
    retrieved_chunks: Sequence[Dict],
    ranked_scores: Sequence[Tuple[int, float]],
    relation_score: float,
) -> str:
    choices_text = "\n".join(f"{key}. {value}" for key, value in choices.items())
    context = build_context(retrieved_chunks)
    candidate_lines = "\n".join(
        f"- ตัวเลือก {choice_id}: score={score:.2f}" for choice_id, score in ranked_scores[:3]
    )
    return (
        ("คำถามนี้ดูเกี่ยวข้องกับฟ้าใหม่\n\n" if relation_score >= 1.2 else "คำถามนี้อาจไม่เกี่ยวข้องกับฟ้าใหม่ ต้องระวังข้อ 10\n\n")
        +
        f"บริบทจากฐานข้อมูล:\n{context}\n\n"
        f"คำถาม:\n{question}\n\n"
        f"ตัวเลือก:\n{choices_text}\n\n"
        f"ตัวเลือกที่ระบบดึงหลักฐานได้ดี:\n{candidate_lines or '- ยังไม่มีตัวเลือกเด่น'}\n\n"
        "พิจารณาหลักฐานที่มีจริงในบริบทก่อนตอบเสมอ "
        "ถ้าตัวเลือกใดมีบางส่วนผิดหรือเกินจริง ให้ตัดทิ้ง "
        "ถ้าไม่มีหลักฐานพอแต่ยังเป็นเรื่องของฟ้าใหม่ให้ตอบ 9 "
        "ถ้าไม่เกี่ยวข้องกับฟ้าใหม่เลยให้ตอบ 10"
    )


def validate_model_names(model_names: Sequence[str]) -> List[str]:
    resolved = [model_name.strip() for model_name in model_names if model_name.strip()]
    invalid = [model_name for model_name in resolved if model_name not in ALLOWED_THAILLM_MODELS]
    if invalid:
        raise ValueError(
            "Unsupported model(s): "
            + ", ".join(invalid)
            + ". Allowed models: "
            + ", ".join(sorted(ALLOWED_THAILLM_MODELS))
        )
    if not resolved:
        raise ValueError("At least one ThaiLLM model is required.")
    return resolved


def make_chat_client(api_key: Optional[str] = None, base_url: Optional[str] = None) -> OpenAI:
    resolved_key = api_key or os.getenv("THAILLM_API_KEY") or os.getenv("OPENAI_API_KEY")
    resolved_base = base_url or os.getenv("THAILLM_BASE_URL") or os.getenv("OPENAI_BASE_URL")
    if not resolved_key or not resolved_base:
        raise ValueError(
            "Missing chat API configuration. Set THAILLM_API_KEY/THAILLM_BASE_URL or pass --api-key/--api-base-url."
        )
    return OpenAI(api_key=resolved_key, base_url=resolved_base)


def ask_model(
    client: OpenAI,
    messages: Sequence[Dict[str, str]],
    model: str,
    temperature: float = 0.0,
    max_tokens: int = 180,
    max_retries: int = 6,
) -> str:
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=list(messages),
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content.strip()
        except RateLimitError:
            sleep_for = min(2 ** attempt + random.random(), 20)
        except (APIConnectionError, APIStatusError):
            sleep_for = min(2 ** attempt + random.random(), 20)
        time.sleep(sleep_for)
    raise RuntimeError(f"Model API failed after retries: {model}")


def vote_with_models(
    client: OpenAI,
    model_names: Sequence[str],
    question: str,
    choices: Dict[str, str],
    retrieved_chunks: Sequence[Dict],
    ranked_scores: Sequence[Tuple[int, float]],
    relation_score: float,
) -> Tuple[Optional[Dict], List[Dict]]:
    votes: List[Dict] = []
    prompt = build_prompt(question, choices, retrieved_chunks, ranked_scores, relation_score)
    for model_name in model_names:
        raw = ask_model(
            client,
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            model=model_name,
        )
        parsed = parse_json_answer(raw) or {"answer": None, "confidence": None, "reason": "", "raw": raw}
        parsed["model"] = model_name
        votes.append(parsed)

    valid_votes = [vote for vote in votes if vote.get("answer") is not None]
    if not valid_votes:
        return None, votes

    counts = Counter(vote["answer"] for vote in valid_votes)
    best_answer, best_count = counts.most_common(1)[0]
    tied_answers = {answer for answer, count in counts.items() if count == best_count}
    winner = max(
        (vote for vote in valid_votes if vote["answer"] in tied_answers),
        key=lambda vote: vote.get("confidence") if vote.get("confidence") is not None else -1.0,
    )
    if len(tied_answers) == 1:
        winner["answer"] = best_answer
    return winner, votes


def decide_special_answer(
    question: str,
    relation_score: float,
    ranked_scores: Sequence[Tuple[int, float]],
) -> Optional[int]:
    normalized_question = normalize_text(question)
    best_score = ranked_scores[0][1] if ranked_scores else 0.0
    second_score = ranked_scores[1][1] if len(ranked_scores) > 1 else 0.0

    if relation_score <= 0.2 and best_score < 2.1:
        return 10
    if any(term in normalized_question for term in NON_FAHMAI_HINTS) and relation_score < 1.0 and best_score < 2.4:
        return 10
    if relation_score >= 1.0 and best_score < 2.0:
        return 9
    if relation_score >= 1.0 and best_score < 2.8 and (best_score - second_score) < 0.35:
        return 9
    return None


def answer_question(
    client: OpenAI,
    question_row: Dict,
    documents: Sequence[Dict],
    chunks: Sequence[Dict],
    embed_model: SentenceTransformer,
    chunk_embeddings: np.ndarray,
    bm25: BM25Okapi,
    model_names: Sequence[str],
    final_k: int = 8,
) -> Dict:
    relation_score, candidate_doc_ids = question_store_relevance(
        question_row["question"], question_row["choices"], documents
    )
    retrieved_chunks = hybrid_retrieve(
        question_row["question"],
        question_row["choices"],
        documents,
        chunks,
        embed_model,
        chunk_embeddings,
        bm25,
        candidate_doc_ids=candidate_doc_ids,
        final_k=final_k,
    )

    ranked_scores: List[Tuple[int, float]] = []
    for choice_id in range(1, 9):
        score = score_choice_against_chunks(question_row["choices"][str(choice_id)], retrieved_chunks)
        ranked_scores.append((choice_id, score))
    ranked_scores.sort(key=lambda item: item[1], reverse=True)

    heuristic_answer = exact_choice_match(question_row["question"], question_row["choices"], retrieved_chunks)
    special_answer = decide_special_answer(question_row["question"], relation_score, ranked_scores)
    selected_vote, votes = vote_with_models(
        client,
        model_names,
        question_row["question"],
        question_row["choices"],
        retrieved_chunks,
        ranked_scores,
        relation_score,
    )

    parsed = selected_vote or {"answer": None, "confidence": None, "reason": "", "raw": ""}
    if special_answer is not None:
        parsed["answer"] = special_answer
        parsed["reason"] = "special_rule"
    elif heuristic_answer and parsed["answer"] in (None, 9, 10):
        parsed["answer"] = heuristic_answer
        parsed["reason"] = "exact_choice_match"
    elif heuristic_answer and ranked_scores and ranked_scores[0][0] != parsed.get("answer"):
        top_choice, top_score = ranked_scores[0]
        current_score = next((score for choice_id, score in ranked_scores if choice_id == parsed.get("answer")), 0.0)
        if top_score >= 5.8 and top_score - current_score >= 1.25:
            parsed["answer"] = top_choice
            parsed["reason"] = "heuristic_override"

    if parsed["answer"] is None:
        parsed["answer"] = heuristic_answer or special_answer or 9

    return {
        "id": question_row["id"],
        "question": question_row["question"],
        "answer": int(parsed["answer"]),
        "confidence": parsed.get("confidence"),
        "reason": parsed.get("reason", ""),
        "raw_response": parsed.get("raw", ""),
        "relation_score": round(relation_score, 4),
        "top_choice_scores": json.dumps(
            [{"choice_id": choice_id, "score": score} for choice_id, score in ranked_scores[:4]],
            ensure_ascii=False,
        ),
        "model_votes": json.dumps(
            [
                {
                    "model": vote.get("model"),
                    "answer": vote.get("answer"),
                    "confidence": vote.get("confidence"),
                    "reason": vote.get("reason"),
                }
                for vote in votes
            ],
            ensure_ascii=False,
        ),
        "sources": json.dumps([chunk["source"] for chunk in retrieved_chunks], ensure_ascii=False),
    }


def run_pipeline(
    data_dir: Path,
    api_key: Optional[str] = None,
    api_base_url: Optional[str] = None,
    n_questions: int = 100,
    model_names: Optional[Sequence[str]] = None,
    embed_model_name: str = DEFAULT_EMBED_MODEL,
    output_csv: str = "submission.csv",
    diagnostics_csv: Optional[str] = "diagnostics.csv",
) -> pd.DataFrame:
    kb_dir = data_dir / "knowledge_base"
    questions = load_questions(data_dir)
    documents = load_documents(kb_dir)
    chunks = build_chunks(documents)
    bm25, _ = build_bm25(chunks)
    embed_model, chunk_embeddings = build_embeddings(chunks, embed_model_name=embed_model_name)
    resolved_models = validate_model_names(model_names or DEFAULT_MODEL_NAMES)
    client = make_chat_client(api_key=api_key, base_url=api_base_url)

    results: List[Dict] = []
    for row in tqdm(questions[:n_questions], desc="Answering"):
        result = answer_question(
            client,
            row,
            documents,
            chunks,
            embed_model,
            chunk_embeddings,
            bm25,
            model_names=resolved_models,
        )
        results.append(result)
        time.sleep(0.25)

    result_df = pd.DataFrame(results)
    submission = result_df[["id", "answer"]].copy()
    submission.to_csv(output_csv, index=False, encoding="utf-8")

    if diagnostics_csv:
        result_df.to_csv(diagnostics_csv, index=False, encoding="utf-8")

    return result_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the FahMai ThaiLLM RAG pipeline.")
    parser.add_argument("--data-dir", default="data", help="Path to the challenge data directory.")
    parser.add_argument("--api-key", default=None, help="Chat API key. Prefer env vars instead.")
    parser.add_argument("--api-base-url", default=None, help="OpenAI-compatible base URL for the ThaiLLM endpoint.")
    parser.add_argument("--n-questions", type=int, default=100, help="Number of questions to answer.")
    parser.add_argument("--model-name", default=None, help="Single ThaiLLM model id. Deprecated; prefer --model-names.")
    parser.add_argument(
        "--model-names",
        default=",".join(DEFAULT_MODEL_NAMES),
        help="Comma-separated ThaiLLM model ids for voting.",
    )
    parser.add_argument("--embed-model-name", default=DEFAULT_EMBED_MODEL, help="Embedding model name.")
    parser.add_argument("--output-csv", default="submission.csv", help="Where to write the submission CSV.")
    parser.add_argument(
        "--diagnostics-csv",
        default="diagnostics.csv",
        help="Where to write per-question diagnostics.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    model_names = [args.model_name] if args.model_name else args.model_names.split(",")
    frame = run_pipeline(
        data_dir=Path(args.data_dir),
        api_key=args.api_key,
        api_base_url=args.api_base_url,
        n_questions=args.n_questions,
        model_names=model_names,
        embed_model_name=args.embed_model_name,
        output_csv=args.output_csv,
        diagnostics_csv=args.diagnostics_csv,
    )
    print(frame[["id", "answer", "confidence", "reason"]].head(10).to_string(index=False))
    print(f"\nSaved submission to {args.output_csv}")
