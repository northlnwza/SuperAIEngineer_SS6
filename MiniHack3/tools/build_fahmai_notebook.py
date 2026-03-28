import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "FahMai_Typhoon_RAG_Colab.ipynb"
MODULE_PATH = ROOT / "fahmai_typhoon_rag.py"


def lines(text: str):
    return [line + "\n" for line in text.strip("\n").splitlines()]


module_source = MODULE_PATH.read_text(encoding="utf-8")


cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": lines(
            """
            # FahMai Typhoon RAG

            Colab-ready notebook for the FahMai challenge.

            Assumption:
            - You upload this notebook into a workspace that also has the challenge `data/` folder, or you mount a folder where `data/` exists.

            Pipeline:
            1. Install dependencies
            2. Point to the `data/` folder
            3. Build structured chunks from the markdown knowledge base
            4. Run hybrid retrieval (dense + BM25 + title bonus)
            5. Use Typhoon to choose the final answer
            6. Export `submission.csv`
            """
        ),
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": lines(
            """
            !pip install -q openai sentence-transformers pythainlp rank-bm25 pandas numpy tqdm
            """
        ),
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": lines(
            """
            import os
            from pathlib import Path

            try:
                from google.colab import userdata, files
            except ImportError:
                userdata = None
                files = None

            DATA_DIR = Path("/content/MiniHack3/data")
            if not DATA_DIR.exists():
                DATA_DIR = Path("data")

            TYPHOON_MODEL = "typhoon-v2.5-30b-a3b-instruct"
            EMBED_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
            N_QUESTIONS = 100

            api_key = os.getenv("OPENTYPHOON_API_KEY") or os.getenv("TYPHOON_API_KEY")
            if not api_key and userdata is not None:
                for key_name in ("OPENTYPHOON_API_KEY", "TYPHOON_API_KEY"):
                    try:
                        api_key = userdata.get(key_name)
                        if api_key:
                            break
                    except Exception:
                        pass

            if not api_key:
                raise ValueError("Set OPENTYPHOON_API_KEY or TYPHOON_API_KEY before running inference.")

            print("DATA_DIR =", DATA_DIR.resolve())
            print("MODEL =", TYPHOON_MODEL)
            """
        ),
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "from pathlib import Path\n",
            "MODULE_SOURCE = r'''\n",
            module_source,
            "\n'''\n",
            "Path('fahmai_typhoon_rag.py').write_text(MODULE_SOURCE.strip() + '\\n', encoding='utf-8')\n",
            "print('Wrote local helper module: fahmai_typhoon_rag.py')\n",
        ],
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": lines(
            """
            import pandas as pd
            from fahmai_typhoon_rag import (
                answer_question,
                build_bm25,
                build_chunks,
                build_embeddings,
                hybrid_retrieve,
                load_documents,
                load_questions,
                make_typhoon_client,
            )

            questions = load_questions(DATA_DIR)
            documents = load_documents(DATA_DIR / "knowledge_base")
            chunks = build_chunks(documents)
            bm25, _ = build_bm25(chunks)
            embed_model, chunk_embeddings = build_embeddings(chunks, embed_model_name=EMBED_MODEL)
            client = make_typhoon_client(api_key)

            print("questions:", len(questions))
            print("documents:", len(documents))
            print("chunks:", len(chunks))
            """
        ),
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": lines(
            """
            sample = questions[0]
            retrieved = hybrid_retrieve(
                sample["question"],
                chunks,
                embed_model,
                chunk_embeddings,
                bm25,
                final_k=5,
            )

            print("Q1:", sample["question"])
            for idx, chunk in enumerate(retrieved, start=1):
                print(f"\\n[{idx}] {chunk['source']} | {chunk['section']}")
                print(chunk["text"][:500])
            """
        ),
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": lines(
            """
            demo = answer_question(
                client=client,
                question_row=questions[0],
                chunks=chunks,
                embed_model=embed_model,
                chunk_embeddings=chunk_embeddings,
                bm25=bm25,
                model_name=TYPHOON_MODEL,
            )

            demo
            """
        ),
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": lines(
            """
            import time
            from tqdm.auto import tqdm

            results = []
            for row in tqdm(questions[:N_QUESTIONS], desc="Answering"):
                results.append(
                    answer_question(
                        client=client,
                        question_row=row,
                        chunks=chunks,
                        embed_model=embed_model,
                        chunk_embeddings=chunk_embeddings,
                        bm25=bm25,
                        model_name=TYPHOON_MODEL,
                    )
                )
                time.sleep(0.25)

            result_df = pd.DataFrame(results)
            result_df.head()
            """
        ),
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": lines(
            """
            submission = result_df[["id", "answer"]].copy()
            submission.to_csv("submission.csv", index=False, encoding="utf-8")
            result_df.to_csv("diagnostics.csv", index=False, encoding="utf-8")

            print(submission.head())
            print("\\nSaved submission.csv and diagnostics.csv")

            if files is not None:
                files.download("submission.csv")
            """
        ),
    },
]


notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

NOTEBOOK_PATH.write_text(json.dumps(notebook, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"Wrote {NOTEBOOK_PATH}")
