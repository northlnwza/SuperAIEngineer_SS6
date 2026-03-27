import re, time, requests

def ask_llm(messages, api_key, model="typhoon", max_retries=5):
    """Call ThaiLLM API with retry and rate-limit handling.

    Available models: typhoon, openthaigpt, pathumma, kbtg
    """
    url = f"http://thaillm.or.th/api/{model}/v1/chat/completions"
    headers = {"Content-Type": "application/json", "apikey": api_key}
    payload = {
        "model": "/model",
        "messages": messages,
        "max_tokens": 2024,
        "temperature": 0,
    }

    for attempt in range(max_retries):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=120)

            if resp.status_code == 429:
                wait = min(2 ** attempt, 30)
                print(f"  Rate limited, waiting {wait}s...")
                time.sleep(wait)
                continue

            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"].strip()

        except requests.exceptions.RequestException as e:
            wait = 2 ** attempt
            print(f"  Error: {e}, retrying in {wait}s...")
            time.sleep(wait)

    return None


def parse_answer(text):
    """Extract answer number from LLM response."""
    if text is None:
        return None
    # Remove any <think>...</think> blocks (some models do chain-of-thought)
    clean = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    # Look for ANSWER: X pattern
    m = re.search(r"ANSWER:\s*(\d+)", clean)
    if m:
        return int(m.group(1))
    # Fallback: first standalone number 1-10
    for d in re.findall(r"\b(\d{1,2})\b", clean):
        if 1 <= int(d) <= 10:
            return int(d)
    return None