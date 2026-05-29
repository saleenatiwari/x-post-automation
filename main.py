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
    """Fetch the 5 most recent entries from the Ember Internship Log."""
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
    """Pull plain text from a Notion rich_text or title property."""
    prop = page["properties"].get(field, {})
    prop_type = prop.get("type")
    if prop_type in ("rich_text", "title"):
        parts = prop.get(prop_type, [])
        return " ".join(p.get("plain_text", "") for p in parts).strip()
    return ""


def parse_entry(page):
    """Return a dict of all relevant fields from a Notion page."""
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

SYSTEM_PROMPT = """You are running Saleena Tiwari's daily X post drafting routine.

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
- Do not use too personal emotions and feelings
- Do not post about internal business, client names, or anything confidential
- Focus only on things that are learnable, relatable, or about personal growth and building journey
- If the entry is mostly internal meeting notes, zoom out and post about a broader theme it surfaces

POST FORMAT RULES:
- Max 280 characters per post
- A short 2-3 tweet thread is okay if the idea genuinely needs room
- No internal business, client names, or confidential details
- Correct grammar, capitalisation, and punctuation

DECISION RULE:
Read the most recent entry carefully. Ask yourself: does this entry contain more than one distinct, standalone idea worth posting about?
- If yes: draft up to 3 posts and assign each a suggested send day (e.g. "Post today", "Post tomorrow", "Post Thursday")
- If no: draft 2 variations of the same idea (different angles or formats)
Never force extra posts. Only split if the ideas are genuinely different and each one stands alone.

SCREENSHOT LOGIC:
Look at "What I Did". If the work could have a visual artefact (Figma, code output, dashboard, diagram, UI, prototype, script terminal) — mark that post with a line at the top:
📸 Consider screenshotting: [specific thing to capture]
If the work is meeting-heavy or has no visual, skip the screenshot line entirely.

OUTPUT FORMAT:
Return only the drafts, clearly labelled like:
DRAFT 1 — Post today
[post text]

DRAFT 2 — Post tomorrow
[post text]

No preamble, no explanation after. Just the labelled drafts."""


def build_user_prompt(today, context_entries):
    context_block = ""
    for e in context_entries:
        context_block += f"\n---\nDate: {e['date']}\nWhat I Did: {e['what_i_did']}\nWhat I Learned: {e['learned']}\n"

    return f"""TODAY'S ENTRY (most recent — use this to draft the posts):

Date: {today['date']}
Idea how to post: {today['idea']}
What I Did: {today['what_i_did']}
What I Learned: {today['learned']}
What Went Well: {today['went_well']}
What to Improve: {today['improve']}
Next Steps / To Do: {today['next_steps']}
Personal: {today['personal']}

CONTEXT ENTRIES (entries 2–5, for project arc and running themes only — do not post about these directly):
{context_block}

Now write the drafts following all rules in the system prompt."""


def call_gemini(today, context_entries):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}"
    headers = {"Content-Type": "application/json"}
    full_prompt = SYSTEM_PROMPT + "\n\n" + build_user_prompt(today, context_entries)
    body = {
        "contents": [
            {"role": "user", "parts": [{"text": full_prompt}]}
        ],
        "generationConfig": {"maxOutputTokens": 1000, "temperature": 0.7},
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
        "parse_mode": "HTML",
    }
    response = requests.post(url, json=payload)
    response.raise_for_status()


def split_drafts(raw_text):
    """Split Gemini's output into individual draft messages."""
    drafts = []
    current = []
    for line in raw_text.splitlines():
        if line.startswith("DRAFT ") and current:
            drafts.append("\n".join(current).strip())
            current = [line]
        else:
            current.append(line)
    if current:
        drafts.append("\n".join(current).strip())
    return [d for d in drafts if d]


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

    # Send to Telegram
    send_telegram(f"Today's X drafts for {today_str}.")

    drafts = split_drafts(raw_drafts)
    for draft in drafts:
        send_telegram(draft)

    send_telegram("✏️ https://x.com/SaleenaTiwari")
    print(f"Sent {len(drafts)} draft(s) to Telegram. Done.")


if __name__ == "__main__":
    main()
