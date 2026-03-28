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


def ask_gemini(messages, api_key, model="gemini-3.1-flash-lite-preview", max_retries=5):
    """Call Google Gemini API with retry handling.
    
    Available models: gemini-2.0-flash-lite, gemini-2.0-flash, gemini-1.5-pro
    """
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    headers = {"Content-Type": "application/json"}
    
    # Convert OpenAI-style messages to Gemini format
    contents = []
    system_instruction = None
    
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        
        if role == "system":
            system_instruction = content
        elif role == "user":
            contents.append({"role": "user", "parts": [{"text": content}]})
        elif role == "assistant":
            contents.append({"role": "model", "parts": [{"text": content}]})
    
    payload = {
        "contents": contents,
        "generationConfig": {
            "temperature": 0,
            "maxOutputTokens": 2048,
        }
    }
    
    if system_instruction:
        payload["systemInstruction"] = {"parts": [{"text": system_instruction}]}
    
    for attempt in range(max_retries):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=120)
            
            if resp.status_code == 429:
                wait = min(2 ** attempt, 30)
                print(f"  Rate limited, waiting {wait}s...")
                time.sleep(wait)
                continue
            
            resp.raise_for_status()
            result = resp.json()
            return result["candidates"][0]["content"]["parts"][0]["text"].strip()
        
        except requests.exceptions.RequestException as e:
            wait = 2 ** attempt
            print(f"  Error: {e}, retrying in {wait}s...")
            time.sleep(wait)
        except (KeyError, IndexError) as e:
            print(f"  Parse error: {e}")
            return None
    
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