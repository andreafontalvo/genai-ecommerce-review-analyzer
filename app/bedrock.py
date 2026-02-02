import os
import json
import re
import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("BEDROCK_API_KEY")
URL = "https://bedrock-runtime.us-east-1.amazonaws.com/model/amazon.nova-micro-v1:0/invoke"


HEADERS = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {API_KEY}"
    }

EXTRACTION_PROMPT = """
    You extract pros and cons from reviews.
    Return ONLY valid JSON with keys: pros, cons.
    """

SUMMARY_PROMPT = """
    You are a clothing products review analyser.
    Always answer with Pros, Cons and Recommendations.
    """

def _call(system, user, temp, tokens):
    payload = {
        "messages": [{
        "role": "user",
        "content": [{"text": user}]
        }],
        "system": [{"text": system}],
        "inferenceConfig": {
        "temperature": temp,
        "maxTokens": tokens
        }
        }

    r = requests.post(URL, json=payload, headers=HEADERS)
    r.raise_for_status()

    data = r.json()
    return data["output"]["message"]["content"][0]["text"]

def extract_pros_cons(df):
    blocks = []
    for i, row in df.iterrows():
        blocks.append(
            f"[{i}] Sentiment={row['Sentiment']}\n"
            f"Review: {row['Review Text']}\n"     
            )

    prompt = "\n".join(blocks)
    raw = _call(EXTRACTION_PROMPT, prompt, 0.5, 400)
    clean = re.sub(r"^```json|```$", "", raw, flags=re.S)
    return json.loads(clean)

def summarize(pros, cons):
    user = json.dumps({"pros": pros, "cons": cons})
    return _call(SUMMARY_PROMPT, user, 0.2, 1500)