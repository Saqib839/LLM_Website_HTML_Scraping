#!/usr/bin/env python3
"""
team_page_scraper.py

Reads team page URLs from a CSV (one URL per line, headerless is supported),
scrapes each page, cleans the HTML text, saves raw text files, then passes
them to Llama-3.2-3B for structured extraction.

Usage:
  python team_page_scraper.py --input team_page/sample_input.csv --output output/result_{timestamp}.csv

Steps:
1. Extract HTML text from each URL
2. Clean: remove scripts, styles, comments, and extra whitespace
3. Save raw cleaned text to raw_output folder (named same as website)
4. Pass cleaned text to Llama-3.2-3B LLM and get response
5. Save/append response to output/results_{timestamp}.csv

Notes:
- Uses Llama-3.2-3B from Hugging Face
- No heuristic extraction, pure LLM-based
- Requires transformers and torch libraries
"""
import argparse
import json
import ast
import io
import os
import re
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup, Comment
import csv
from datetime import datetime

try:
    # from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
    from transformers import AutoModelForCausalLM, AutoTokenizer, TextGenerationPipeline
    TF_AVAILABLE = True
except Exception:
    TF_AVAILABLE = False


# Primary headers; some sites block very generic agents. We rotate a few realistic UAs
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_4) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.4 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:120.0) Gecko/20100101 Firefox/120.0",
]

# Extra polite headers
BASE_HEADERS = {
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

# session re-used for requests
SESSION = requests.Session()



def read_urls_from_csv(path):
    urls = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # allow comma-separated or single-column lines
            if "," in line and not line.startswith("http"):
                parts = [p.strip() for p in line.split(",") if p.strip()]
                urls.extend(parts)
            else:
                urls.append(line)
    return urls


def fetch_url(url, timeout=20):
    """Fetch a URL with simple retry and user-agent rotation to reduce 403s."""
    # Try with session + a few UAs
    last_exc = None
    for ua in USER_AGENTS:
        headers = BASE_HEADERS.copy()
        headers["User-Agent"] = ua
        # set Referer to domain root which sometimes helps
        try:
            headers["Referer"] = f"{urlparse(url).scheme}://{urlparse(url).netloc}/"
        except Exception:
            pass
        try:
            r = SESSION.get(url, headers=headers, timeout=timeout)
            r.raise_for_status()
            return r.text
        except requests.exceptions.HTTPError as he:
            last_exc = he
            # if 403 try next UA
            if r is not None and r.status_code == 403:
                # try next UA
                continue
            else:
                print(f"HTTP error fetching {url}: {he}")
                return None
        except Exception as e:
            last_exc = e
            continue
    print(f"Failed to fetch {url}: {last_exc}")
    return None


def clean_text_content(html_text):
    """
    Extract and clean HTML text:
    1. Remove scripts, styles, comments
    2. Remove language keywords and extra whitespace
    3. Return clean text content
    """
    soup = BeautifulSoup(html_text, "html.parser")
    
    # Remove scripts/styles and comments
    for tag in soup(["script", "style", "noscript", "template", "meta", "link"]):
        tag.decompose()
    
    # Remove HTML comments
    for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
        comment.extract()
    
    # Get text content
    texts = []
    for elem in soup.find_all(string=True):
        txt = elem.strip()
        if not txt:
            continue
        # Skip tiny fragments
        if len(txt) < 2:
            continue
        texts.append(txt)
    
    # Join and clean excessive whitespace
    content = " ".join(texts)
    
    # Remove language keywords and programming-related noise
    noise_patterns = [
        r"\bjavascript\b",
        r"\bjquery\b",
        r"\bcss\b",
        r"\bhtml\b",
        r"\bxml\b",
        r"\bvar\b",
        r"\bfunction\b",
        r"\bwindow\b",
        r"\bdocument\b",
        r"\bcookie\b",
        r"\bsession\b",
        r"\btitle\b\s*[>:]",
    ]
    
    for pattern in noise_patterns:
        content = re.sub(pattern, "", content, flags=re.IGNORECASE)
    
    # Collapse multiple spaces/newlines
    content = re.sub(r"\s+", " ", content)
    content = content.strip()
    
    return content


def extract_site_documents(url):
    """
    Extract and clean HTML text from URL.
    Returns cleaned text content only (no profile link following).
    """
    html = fetch_url(url)
    if not html:
        return None
    
    # Clean the HTML and extract text
    cleaned_text = clean_text_content(html)
    
    return {
        "page_url": url,
        "page_text": cleaned_text
    }


def init_llm(model_name="meta-llama/Llama-3.2-3B-Instruct"):
    """Initialize Llama-3.2-3B model for text generation."""
    if not TF_AVAILABLE:
        return None
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto"
        )
        pipe = TextGenerationPipeline(model=model, tokenizer=tokenizer)
        return pipe
    except Exception as e:
        print(f"Failed to initialize model {model_name}: {e}")
        return None


def ask_llm_for_extraction(pipe, cleaned_text, website_url):
    """
    Pass cleaned text to Llama-3.2-3B and request structured CSV extraction.
    Returns a list of dicts matching: website, full_name, full_bio, age, hometown, education, experience, photo_url
    """
    if not pipe:
        return None
    
    # Load prompt template
    try:
        demo_prompt = open("demo_prompt.txt", "r", encoding="utf-8").read()
    except Exception:
        demo_prompt = (
                "Extract ALL doctors from the text below.\n"
                "Return ONLY a valid JSON array.\n"
                "Each object must include:\n"
                " full_name, full_bio, age, hometown, education, experience, photo_url\n"
                "If missing, use empty string.\n"
            )

    # Prepare the prompt with cleaned text
    prompt = f"""{demo_prompt}
            TEXT TO EXTRACT FROM:
            {cleaned_text}
            """

    try:
        # Generate response from Llama (expecting JSON)
        outputs = pipe(
            prompt,
            max_new_tokens=2048,
            num_return_sequences=1,
            do_sample=False,
            temperature=0.0
        )

        # Extract generated text
        if isinstance(outputs, list) and len(outputs) > 0:
            text = outputs[0].get("generated_text", "")
            if text is None:
                text = ""
            # If the model echoes the prompt, strip it
            if isinstance(text, str) and text.startswith(prompt):
                text = text[len(prompt):]
        else:
            text = str(outputs)

        # Save raw LLM output for debugging
        try:
            safe_site = re.sub(r"[^0-9a-zA-Z]+", "_", str(website_url or "site"))
            # raw_dir = os.path.join("output", "llm_raw")
            raw_dir = "llm_raw_output"
            os.makedirs(raw_dir, exist_ok=True)
            raw_path = os.path.join(raw_dir, f"{safe_site}.txt")
            with open(raw_path, "w", encoding="utf-8") as rf:
                rf.write(text or "")
        except Exception:
            pass

        # Helper: try to extract JSON substring from text
        def _find_json_substring(s):
            if not s or not isinstance(s, str):
                return None
            s = s.strip()
            # Prefer array
            start = s.find('[')
            if start != -1:
                # try to find a matching closing bracket by last ']' occurrence
                end = s.rfind(']')
                if end != -1 and end > start:
                    return s[start:end+1]
            # fallback to object
            start = s.find('{')
            if start != -1:
                end = s.rfind('}')
                if end != -1 and end > start:
                    return s[start:end+1]
            return None

        json_sub = _find_json_substring(text)
        people = []
        if json_sub:
            try:
                parsed = json.loads(json_sub)
            except Exception:
                parsed = None
        else:
            parsed = None

        if parsed is None:
            # Try a looser JSON load on whole text
            try:
                parsed = json.loads(text)
            except Exception:
                parsed = None

        if parsed is None:
            print(f"LLM did not return valid JSON for {website_url}")
            return None

        # Normalize to list
        if isinstance(parsed, dict):
            parsed = [parsed]

        expected_keys = ["full_name", "full_bio", "age", "hometown", "education", "experience", "photo_url"]
        for entry in parsed:
            if not isinstance(entry, dict):
                continue
            person = {k: str(entry.get(k, "")).strip() if entry.get(k) is not None else "" for k in expected_keys}
            # Add website from the caller
            person["website"] = entry.get("website") or website_url
            people.append(person)

        if people:
            return people

        print(f"LLM returned JSON but no valid person entries for {website_url}")
        return None

    except Exception as e:
        print(f"LLM generation failed for {website_url}: {e}")
        return None


# def save_output_item(output_path, item):
#     os.makedirs(os.path.dirname(output_path), exist_ok=True)
#     # append jsonl line
#     with open(output_path, "a", encoding="utf-8") as f:
#         f.write(json.dumps(item, ensure_ascii=False) + "\n")


def write_csv_header(path, fieldnames):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

def append_csv_row(path, fieldnames, row):
    with open(path, "a", encoding="utf-8", newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writerow(row)


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="team_page/sample_input.csv")
    parser.add_argument("--output", default=f"output/result_{timestamp}.csv")
    parser.add_argument("--model", default="meta-llama/Llama-3.2-3B-Instruct", help="Model name from Hugging Face Hub")
    args = parser.parse_args()

    urls = read_urls_from_csv(args.input)
    print(f"Found {len(urls)} URLs in {args.input}")
    
    # Initialize LLM
    pipe = init_llm(args.model)
    if pipe is None:
        print(f"ERROR: Could not initialize LLM model {args.model}")
        print("Make sure the model is available on Hugging Face Hub and you have sufficient GPU memory.")
        return

    # CSV output configuration
    csv_fieldnames = ["website", "full_name", "full_bio", "age", "hometown", "education", "experience", "photo_url"]
    write_csv_header(args.output, csv_fieldnames)

    seen_people = {}  # Track people by (website, name) to avoid duplicates
    
    for url in urls:
        print(f"\n{'='*80}")
        print(f"Processing: {url}")
        print('='*80)
        
        # Step 1: Extract HTML text
        site_doc = extract_site_documents(url)
        if not site_doc:
            print(f"❌ Failed to extract content for {url}")
            append_csv_row(args.output, csv_fieldnames, {
                "website": url,
                "full_name": "",
                "full_bio": "",
                "age": "",
                "hometown": "",
                "education": "",
                "experience": "",
                "photo_url": ""
            })
            continue
        
        cleaned_text = site_doc.get("page_text", "")
        print(f"✓ Extracted text: {len(cleaned_text)} characters")
        
        # Step 2 & 3: Already done in extract_site_documents (cleaning and returning cleaned text)
        # Step 3b: Save raw cleaned text to file
        safe_site = re.sub(r"[^0-9a-zA-Z]+", "_", urlparse(url).netloc + urlparse(url).path)
        raw_path = os.path.join("raw_html_text", f"{safe_site}.txt")
        os.makedirs(os.path.dirname(raw_path), exist_ok=True)
        
        with open(raw_path, "w", encoding="utf-8") as f:
            f.write(f"Website: {url}\n")
            # f.write(f"Cleaned Text:\n")
            f.write("="*80 + "\n")
            f.write(cleaned_text)
        
        print(f"✓ Saved raw cleaned text to: {raw_path}")
        
        # Step 4: Pass to Llama-3.2-3B
        print(f"🔄 Querying Llama-3.2-3B for extraction...")
        people = ask_llm_for_extraction(pipe, cleaned_text, url)
        
        if not people:
            print(f"⚠ LLM returned no people for {url}")
            append_csv_row(args.output, csv_fieldnames, {
                "website": url,
                "full_name": "",
                "full_bio": "",
                "age": "",
                "hometown": "",
                "education": "",
                "experience": "",
                "photo_url": ""
            })
            continue
        
        print(f"✓ Extracted {len(people)} person/people")
        
        # Step 5: Save/append to CSV
        wrote_count = 0
        for person in people:
            person_key = (url, person.get("full_name", "").strip())
            if person_key in seen_people:
                continue
            seen_people[person_key] = True
            
            row = {
                "website": person.get("website", url),
                "full_name": person.get("full_name", ""),
                "full_bio": person.get("full_bio", ""),
                "age": person.get("age", ""),
                "hometown": person.get("hometown", ""),
                "education": person.get("education", ""),
                "experience": person.get("experience", ""),
                "photo_url": person.get("photo_url", "")
            }
            append_csv_row(args.output, csv_fieldnames, row)
            wrote_count += 1
        
        if wrote_count > 0:
            print(f"✓ Wrote {wrote_count} row(s) to CSV")
        else:
            print(f"⚠ No valid rows written")

    print(f"\n{'='*80}")
    print(f"✓ Done! Results saved to: {args.output}")
    print(f"✓ Raw cleaned texts saved to: raw_output/")
    print('='*80)


if __name__ == "__main__":
    main()
