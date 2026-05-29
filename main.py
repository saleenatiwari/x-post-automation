import os
import time
import requests
from datetime import datetime

# ── Config from environment variables (set in GitHub Secrets) ──────────────
NOTION_TOKEN = os.environ["NOTION_TOKEN"]
NOTION_DATABASE_ID = os.environ["NOTION_DATABASE_ID"]
GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
TELEGRAM_BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
TELEGRAM_CHAT_ID = os.environ["TELEGRAM_CHAT_ID"]

NOTION_VERSION = "2022-06-28"


# ── 1. Read Notion ─────────────────────────────────────────────────────────

def get_notion_entries():
    url = f"https://api.notion.com/v1/databases/{NOTION_DATABASE_ID}/query"
    headers = {
        "Authorization": f"Bearer {NOTION_TOKEN}",
        "Notion-Version": NOTION_VERSION,
        "Content-Type": "application/json",
    }
    body = {
        "sorts": [{"property": "Date", "direction": "descending"}],
        "page_size": 5,
    }
    response = requests.post(url, headers=headers, json=body)
    response.raise_for_status()
    return response.json()["results"]


def extract_text(page, field):
    prop = page["properties"].get(field, {})
    prop_type = prop.get("type")
    if prop_type in ("rich_text", "title"):
        parts = prop.get(prop_type, [])
        return " ".join(p.get("plain_text", "") for p in parts).strip()
    return ""


def parse_entry(page):
    return {
        "date":       extract_text(page, "Date"),
        "idea":       extract_text(page, "Idea how to post"),
        "what_i_did": extract_text(page, "What I Did"),
        "learned":    extract_text(page, "What I Learned"),
        "went_well":  extract_text(page, "What Went Well"),
        "improve":    extract_text(page, "What to Improve"),
        "next_steps": extract_text(page, "Next Steps / To Do"),
        "personal":   extract_text(page, "Personal"),
    }


# ── 2. Call Gemini ─────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are drafting X (Twitter) posts for Saleena Tiwari.

ABOUT SALEENA:
- First-year Honors CS student at Georgia Tech, co-founder of AI startup Cohort
- Interning at Ember, building in public on X
- Audience: early-stage builders, women in tech, people just starting out (most under 1K followers)

VOICE RULES:
- Casual, stream-of-consciousness, direct, self-aware
- Occasionally dry humor
- Never LinkedIn polish — no "excited to share", no "thrilled to announce"
- Short sentences. Real feelings. Specific details beat vague inspiration
- Frame from personal experience or behaviour first, then the insight
- End each post with a question, a hot take, or an honest admission
- No hashtags unless completely natural. No emojis unless they genuinely add something
- Never sound like an ad for herself
- Do not use overly personal emotions or feelings
- Do not post about internal business, client names, or anything confidential
- Focus only on things that are learnable, relatable, or about personal growth and the building journey
- If the entry is mostly internal meeting notes, zoom out to a broader theme it surfaces

POST FORMAT RULES:
- Max 280 characters per post
- A short 2-3 tweet thread is okay if the idea genuinely needs room — format as: "1/ text\n\n2/ text\n\n3/ text"
- No internal business details, client names, or confidential info
- Correct grammar, capitalisation, and punctuation
- Each post must be fully written out — not a placeholder, not a summary, not a title. The actual post text, ready to copy and paste.

DECISION RULE:
Does today's entry contain more than one distinct standalone idea worth posting about?
- If yes: draft up to 3 posts, each with a different idea, assign a send day
- If no: draft 2 variations of the same idea with different angles or formats
Never force extra posts. Only split if the ideas are genuinely different and each stands alone.

SCREENSHOT RULE:
Only add a screenshot suggestion if the work produced a clear visual artefact (Figma file, code terminal output, dashboard, UI, diagram). If the day was meeting-heavy or has no visual output, skip it entirely. Do NOT suggest screenshotting a journal entry or a diagram of an automation — only real work outputs.

OUTPUT FORMAT — follow this exactly, no deviation:
Each draft must be separated by the marker ---DRAFT--- on its own line.
Start each draft with the send day label on its own line, like: Post today
Then a blank line.
Then the full post text, ready to copy and paste.
If there is a screenshot suggestion, put it on its own line at the very top before the send day, starting with: 📸

Example output:
📸 Consider screenshotting: [specific real artefact]
Post today

[full post text here, written out completely]
---DRAFT---
Post tomorrow

[full post text here, written out completely]

Do not include any intro, explanation, preamble, or closing text. Just the drafts separated by ---DRAFT---."""


def build_user_prompt(today, context_entries):
    context_block = ""
    for e in context_entries:
        context_block += f"\n---\nDate: {e['date']}\nWhat I Did: {e['what_i_did']}\nWhat I Learned: {e['learned']}\n"

    return f"""TODAY'S ENTRY — use this to write the posts:

Date: {today['date']}
Idea how to post: {today['idea']}
What I Did: {today['what_i_did']}
What I Learned: {today['learned']}
What Went Well: {today['went_well']}
What to Improve: {today['improve']}
Next Steps / To Do: {today['next_steps']}
Personal: {today['personal']}

CONTEXT (entries 2–5, understand the arc only — do not post about these):
{context_block}

Write the drafts now. Full post text only. No placeholders."""


def call_gemini(today, context_entries):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={GEMINI_API_KEY}"
    headers = {"Content-Type": "application/json"}
    full_prompt = SYSTEM_PROMPT + "\n\n" + build_user_prompt(today, context_entries)
    body = {
        "contents": [
            {"role": "user", "parts": [{"text": full_prompt}]}
        ],
        "generationConfig": {"maxOutputTokens": 1500, "temperature": 0.9},
    }
    for attempt in range(3):
        response = requests.post(url, headers=headers, json=body)
        if response.status_code == 429:
            print(f"Rate limited, waiting 30s (attempt {attempt + 1}/3)...")
            time.sleep(30)
            continue
        response.raise_for_status()
        return response.json()["candidates"][0]["content"]["parts"][0]["text"]
    response.raise_for_status()


# ── 3. Send to Telegram ────────────────────────────────────────────────────

def send_telegram(text):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": text,
    }
    response = requests.post(url, json=payload)
    response.raise_for_status()


def split_drafts(raw_text):
    """Split on the ---DRAFT--- marker and clean each draft."""
    parts = raw_text.split("---DRAFT---")
    drafts = []
    for part in parts:
        cleaned = part.strip()
        if cleaned:
            drafts.append(cleaned)
    return drafts


# ── 4. Main ────────────────────────────────────────────────────────────────

def main():
    today_str = datetime.now().strftime("%A, %B %-d")
    print(f"Running for {today_str}...")

    # Read Notion
    pages = get_notion_entries()
    entries = [parse_entry(p) for p in pages]
    today_entry = entries[0]
    context_entries = entries[1:]
    print(f"Most recent entry: {today_entry['date']}")

    # Call Gemini
    print("Calling Gemini...")
    raw_drafts = call_gemini(today_entry, context_entries)
    print("Gemini response received.")
    print("--- RAW OUTPUT ---")
    print(raw_drafts)
    print("--- END RAW OUTPUT ---")

    # Send intro
    send_telegram(f"X drafts for {today_str} ✏️")

    # Send each draft as its own message
    drafts = split_drafts(raw_drafts)
    for i, draft in enumerate(drafts, 1):
        send_telegram(draft)
        print(f"Sent draft {i}.")

    # Send link
    send_telegram("https://x.com/SaleenaTiwari")
    print(f"Done. Sent {len(drafts)} draft(s) to Telegram.")


if __name__ == "__main__":
    main()
