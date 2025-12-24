import os
import json
import math
import requests
import pandas as pd
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from openai import OpenAI

# ---------------- CONFIG ----------------
INPUT_CSV = "input.csv"
OUTPUT_CSV = "output.csv"
WEBSITE_CONTENT_COLUMN = "website_content"

BATCH_SIZE = 5
MAX_WEBSITE_CHARS = 12000
MODEL = "gpt-4o-mini"

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ---------------- SCRAPER ----------------
def fetch_website_text(url):
    try:
        r = requests.get(
            url,
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=15
        )
        soup = BeautifulSoup(r.text, "html.parser")

        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()

        text = " ".join(soup.stripped_strings)
        return text[:MAX_WEBSITE_CHARS]
    except Exception as e:
        print(f"[WARN] {url} scrape failed: {e}")
        return ""

# ---------------- BATCH OPENAI CALL ----------------
def enrich_batch(rows, user_prompt):
    payload = []

    for r in rows:
        payload.append({
            "row_id": r["row_id"],
            "row_data": r["row_data"],
            "website_content": r["website_content"]
        })

    prompt = f"""
You are a data extraction assistant.

For EACH item in the list below:
- Use ONLY the website_content
- Fill missing fields
- Keep existing values unchanged
- If data not found, use empty string

User instructions:
{user_prompt}

Return STRICT JSON ARRAY.
Each object MUST include row_id and all row_data fields.

Input:
{json.dumps(payload, indent=2)}
"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You extract structured data from websites."},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )

    return json.loads(response.choices[0].message.content)

# ---------------- MAIN ----------------
def main():
    df = pd.read_csv(INPUT_CSV)

    if WEBSITE_CONTENT_COLUMN not in df.columns:
        df[WEBSITE_CONTENT_COLUMN] = ""

    USER_PROMPT = """
Fields to fill:
- full_name
- full_bio
- designation (Owner or Associate)
- graduation_year
- age (26 + current_year - graduation_year)

"""

    records = []

    # scrape websites first
    for idx, row in df.iterrows():
        website = str(row.iloc[0]).strip()
        print(f"[SCRAPE] {website}")

        website_text = fetch_website_text(website)
        row_dict = row.to_dict()
        row_dict[WEBSITE_CONTENT_COLUMN] = website_text

        records.append({
            "row_id": idx,
            "row_data": row_dict,
            "website_content": website_text
        })

    # batch processing
    enriched_map = {}

    total_batches = math.ceil(len(records) / BATCH_SIZE)

    for i in range(0, len(records), BATCH_SIZE):
        batch = records[i:i + BATCH_SIZE]
        batch_no = (i // BATCH_SIZE) + 1
        print(f"[OPENAI] Batch {batch_no}/{total_batches}")

        try:
            enriched = enrich_batch(batch, USER_PROMPT)
            for item in enriched:
                enriched_map[item["row_id"]] = item
        except Exception as e:
            print(f"[ERROR] Batch failed: {e}")
            for r in batch:
                enriched_map[r["row_id"]] = {
                    "row_id": r["row_id"],
                    **r["row_data"]
                }

    # rebuild dataframe
    final_rows = []
    for idx in range(len(df)):
        final_rows.append(enriched_map[idx])

    out_df = pd.DataFrame(final_rows)
    out_df.to_csv(OUTPUT_CSV, index=False)

    print(f"[DONE] Saved → {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
