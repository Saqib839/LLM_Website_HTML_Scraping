#!/usr/bin/env python3
"""
team_page_scraper.py

Reads team page URLs from a CSV (one URL per line, headerless is supported),
scrapes each page and any linked profile/sub-pages for individual doctors/staff,
then uses a free LLM (default: Hugging Face `google/flan-t5-small`) to request
a structured JSON response for the site's people data.

Usage:
  python team_page_scraper.py --input team_page/sample_input.csv --output output/team_pages_results.jsonl

Notes:
- This script tries to use `transformers` + `google/flan-t5-small` which is free
  via Hugging Face model hub. The model will be downloaded automatically if
  internet access is available. If the model or transformers is unavailable,
  the script falls back to saving raw extracted text files under `output/`.
"""
import argparse
import json
import os
import re
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup, Comment
import csv
from datetime import datetime

try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
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


def visible_text_from_soup(soup):
    # Remove scripts/styles and comments
    for tag in soup(["script", "style", "noscript", "template"]):
        tag.decompose()
    for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
        comment.extract()
    texts = []
    # Use string=True per deprecation guidance
    for elem in soup.find_all(string=True):
        txt = elem.strip()
        if not txt:
            continue
        # Skip tiny fragments
        if len(txt) < 2:
            continue
        texts.append(txt)
    return "\n".join(texts)


def find_profile_links(soup, base_url):
    links = set()
    candidates = []
    # Heuristic patterns
    patterns = [r"doctor", r"dr\b", r"team", r"staff", r"profile", r"provider", r"physician", r"bio", r"practitioner"]
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if href.startswith("mailto:"):
            continue
        full = urljoin(base_url, href)
        text = (a.get_text(" ") or "").lower()
        # check anchor text or href for patterns
        for p in patterns:
            if re.search(p, href, re.I) or re.search(p, text, re.I):
                candidates.append(full)
                break
    # also try to find links inside blocks with class names
    for cls in ["team", "staff", "doctors", "providers", "profiles", "physicians"]:
        # Use a safe CSS class selector (e.g. .team) and fallback to attribute scan
        try:
            for block in soup.select(f".{cls}"):
                for a in block.find_all("a", href=True):
                    links.add(urljoin(base_url, a["href"]))
        except Exception:
            # Fallback: find elements whose class attribute contains the token
            for block in soup.find_all(attrs={"class": lambda v: v and cls in " ".join(v) if isinstance(v, (list, tuple)) else (cls in v)}):
                for a in block.find_all("a", href=True):
                    links.add(urljoin(base_url, a["href"]))
    links.update(candidates)
    # Filter to same domain and remove duplicates
    base_net = urlparse(base_url).netloc
    filtered = []
    for u in links:
        try:
            if urlparse(u).netloc and urlparse(u).netloc != base_net:
                # keep external if it's clearly a profile page? skip for now
                continue
        except Exception:
            continue
        filtered.append(u)
    # dedupe while preserving order
    seen = set()
    out = []
    for u in filtered:
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


def extract_site_documents(url):
    html = fetch_url(url)
    if not html:
        return None
    soup = BeautifulSoup(html, "html.parser")
    main_text = visible_text_from_soup(soup)
    profiles = find_profile_links(soup, url)
    profile_texts = []
    for p in profiles:
        if p == url:
            continue
        html_p = fetch_url(p)
        if not html_p:
            continue
        soup_p = BeautifulSoup(html_p, "html.parser")
        profile_texts.append({"url": p, "text": visible_text_from_soup(soup_p)})
    return {"page_url": url, "page_text": main_text, "profiles": profile_texts}


def init_llm(model_name="google/flan-t5-small"):
    if not TF_AVAILABLE:
        return None
    try:
        # small model to keep resource needs low
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer)
        return pipe
    except Exception as e:
        print(f"Failed to initialize model {model_name}: {e}")
        return None


def ask_llm_for_json(pipe, site_doc):
    # Build prompt
    combined = []
    combined.append(f"PAGE_URL: {site_doc['page_url']}")
    combined.append("PAGE_TEXT:\n" + site_doc["page_text"][:4000])
    for i, p in enumerate(site_doc["profiles"]):
        combined.append(f"PROFILE_URL: {p['url']}")
        combined.append("PROFILE_TEXT:\n" + p["text"][:2000])
    prompt = (
        "Extract structured information about the people (doctors/staff/providers) found in the following website text. "
        "Return a single valid JSON array. Each item should be an object with these fields: name, full_bio, age, hometown, education, experience, photo_url. "
        "If a field is not present set it to null. Use simple strings or arrays for fields. Do not return any extra commentary, only valid JSON.\n\n"
        + "\n\n".join(combined)
    )
    # Flan-T5 benefits from short inputs; we truncated above. Now run generation
    try:
        out = pipe(prompt, max_length=1024, do_sample=False)
        text = out[0]["generated_text"]
        # Try to find JSON substring
        m = re.search(r"(\[\s*\{[\s\S]*\}\s*\])", text)
        if m:
            candidate = m.group(1)
        else:
            candidate = text.strip()
        # Ensure it's valid JSON
        try:
            parsed = json.loads(candidate)
            return parsed
        except Exception:
            # Try to be more forgiving: replace single quotes
            candidate2 = candidate.replace("'", '"')
            try:
                return json.loads(candidate2)
            except Exception:
                print("LLM output could not be parsed as JSON. Returning raw output.")
                return {"llm_raw": text}
    except Exception as e:
        print(f"LLM generation failed: {e}")
        return None


def save_output_item(output_path, item):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    # append jsonl line
    with open(output_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")


def infer_age_from_text(text, current_year=None):
    # Find graduation year in text (look for 19xx or 20xx)
    if not text:
        return ""
    current_year = current_year or datetime.now().year
    # Capture birth year mentions like "born in 1960" or graduation "1982"
    birth_year_match = re.search(r"born.*?(\d{4})|age\s+(\d+)", text, re.IGNORECASE)
    if birth_year_match:
        if birth_year_match.group(1):  # born in YYYY
            year_int = int(birth_year_match.group(1))
            if 1920 < year_int < current_year:
                return str(current_year - year_int)
        elif birth_year_match.group(2):  # age N
            return birth_year_match.group(2)
    
    # Fall back to graduation year (students graduate at ~26 after dental school)
    matches = re.findall(r"(19\d{2}|20\d{2})", text)
    grad_year = None
    for year in matches:
        year_int = int(year)
        if 1950 < year_int <= current_year:  # plausible grad years
            grad_year = year_int
            break
    if grad_year:
        return str(26 + (current_year - grad_year))
    return ""


def write_csv_header(path, fieldnames):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

def append_csv_row(path, fieldnames, row):
    with open(path, "a", encoding="utf-8", newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writerow(row)


def heuristic_extract_people(site_doc):
    """
    Fallback extractor when the LLM output is unavailable or unparseable.
    Tries to split the visible text into doctor-sized chunks based on 'Dr.' lines.
    Consolidates all text for one person into a single record.
    """
    # Combine main page text + any profile texts
    texts = [site_doc.get("page_text") or ""]
    for p in site_doc.get("profiles", []):
        texts.append(p.get("text") or "")
    combined = "\n".join(texts)
    
    # Remove common section headers, footers, and noise
    noise_patterns = [
        r"(Professional Info|Volunteer Work|Personal Info|Designed by ECMWORX|More\n|Hygiene Team|Dental Assisting Team|Business Team|About Us|Links & Resources|html|Meet the Staff)",
        r"^(Conveniently Located|Metro Access|Metrobus|Convenient and Flexible Office Hours|Contact|Services|Office|Patient Resources|Testimonials|Links|New Patients).*?(?=\n|$)",
        r"^(©|Find us on|Go to Top|Skip to content|Site by|10162)",
        r"(Facebook page|Mail page|Follow|Social media|©.*?Rights Reserved)",
    ]
    
    for pattern in noise_patterns:
        combined = re.sub(pattern, "", combined, flags=re.MULTILINE | re.IGNORECASE)
    
    lines = [ln.strip() for ln in combined.splitlines() if ln.strip() and len(ln.strip()) > 2]

    people = []
    current_person = None
    skip_until_next_doctor = False
    pending_dr_first = None  # Track "Dr. FirstName" waiting for LastName
    
    for i, line in enumerate(lines):
        # Check if this is a section header for non-doctor staff (single names)
        if re.match(r"^(Heidi|Kristin|Gena|Cherry|Amy|Kelly|Erika|Madonna|Cori|Corinna|Leah|Jodi|Kim)$", line, re.IGNORECASE):
            skip_until_next_doctor = True
            pending_dr_first = None
            continue
        
        # Check for "Dr. FirstName" followed by LastName on next line
        # Pattern 1: Single line with full name "Dr. FirstName LastName"
        is_doctor_full = (
            re.match(r"^\s*Dr\.\s+[\w\.]+\s+[\w\.]+\s*$", line) and
            len(line) < 100
        )
        
        # Pattern 2: Line starting with "Dr. FirstName" (waiting for next line to be LastName)
        dr_first_match = re.match(r"^\s*Dr\.\s+([\w\.]+)\s*$", line)
        
        # Pattern 3: If we have pending "Dr. FirstName", check if this line is a LastName
        is_last_name = (
            pending_dr_first and 
            re.match(r"^[A-Z][a-z]+$", line) and
            len(line) < 50
        )
        
        if is_last_name:
            # Combine pending "Dr. FirstName" + this "LastName"
            full_name = f"{pending_dr_first} {line}"
            skip_until_next_doctor = False
            
            # Save previous person if substantial
            if current_person and len(current_person.get("bio", "")) > 20:
                people.append(current_person)
            
            current_person = {
                "name": full_name,
                "bio": "",
                "education": "",
                "experience": "",
                "hometown": "",
                "age": "",
                "photo_url": ""
            }
            pending_dr_first = None
        elif is_doctor_full:
            # Full name on single line
            skip_until_next_doctor = False
            
            # Save previous person if substantial
            if current_person and len(current_person.get("bio", "")) > 20:
                people.append(current_person)
            
            name = re.sub(r'\s+', ' ', line).strip()
            current_person = {
                "name": name,
                "bio": "",
                "education": "",
                "experience": "",
                "hometown": "",
                "age": "",
                "photo_url": ""
            }
            pending_dr_first = None
        elif dr_first_match:
            # "Dr. FirstName" on this line, might be followed by LastName
            pending_dr_first = line.strip()
        elif current_person and not skip_until_next_doctor and not pending_dr_first:
            # Add line to current doctor's bio
            current_person["bio"] += (" " if current_person["bio"] else "") + line
            
            # Extract education
            if re.search(r"(University|College|School of Dentistry|Dental School|Cum Laude|graduated|degree|DDS|DMD|B\.S\.|B\.A\.)", line, re.IGNORECASE):
                current_person["education"] += (" | " if current_person["education"] else "") + line
            
            # Extract experience
            if re.search(r"(\d+\s+years?|since \d{4}|residency|practice|practicing|joined|member|Captain|private practice|board certified)", line, re.IGNORECASE):
                current_person["experience"] += (" | " if current_person["experience"] else "") + line
            
            # Extract hometown
            if re.search(r"(raised in|born in|area of|from|hometown|native of|grew up)", line, re.IGNORECASE):
                current_person["hometown"] += (" | " if current_person["hometown"] else "") + line
    
    # Save last person if substantial
    if current_person and len(current_person.get("bio", "")) > 20:
        people.append(current_person)
    
    return people

def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="team_page/sample_input.csv")
    parser.add_argument("--output", default=f"output/result_{timestamp}.csv")
    parser.add_argument("--model", default="google/flan-t5-small", help="Transformers model name or 'none' to skip LLM step")
    parser.add_argument("--save-raw-only", action="store_true", help="Do not call LLM, only save extracted raw text")
    args = parser.parse_args()

    urls = read_urls_from_csv(args.input)
    print(f"Found {len(urls)} URLs in {args.input}")
    pipe = None
    if not args.save_raw_only and args.model.lower() != "none":
        pipe = init_llm(args.model)
        if pipe is None:
            print("LLM unavailable; continuing in raw-only mode.")

    # Always output CSV
    csv_fieldnames = ["website", "full_name", "full_bio", "age", "hometown", "education", "experience", "photo_url"]
    write_csv_header(args.output, csv_fieldnames)

    seen_people = {}  # Track people by (website, name) to avoid duplicates
    
    for url in urls:
        print(f"Processing {url}")
        site_doc = extract_site_documents(url)
        if not site_doc:
            continue
        people = []
        if pipe and not args.save_raw_only:
            llm_result = ask_llm_for_json(pipe, site_doc)
            if isinstance(llm_result, list):
                people = llm_result
        # Fallback: if no LLM people found, use heuristic extraction from visible text
        if not people:
            people = heuristic_extract_people(site_doc)
            # If we *still* have no people, emit a single empty record to keep the site tracked
            if not people:
                people = [{}]
        
        for person in people:
            # Compose row with the required columns
            bio = person.get("bio", "") if person else ""
            education = person.get("education", "") if person else ""
            experience = person.get("experience", "") if person else ""
            photo_url = person.get("photo_url", "") if person and "photo_url" in person else ""
            full_name = person.get("name", "") if person else ""
            
            # Skip duplicate entries (same website + name combination)
            person_key = (url, full_name.strip())
            if person_key in seen_people:
                continue
            seen_people[person_key] = True
            
            row = {
                "website": url,
                "full_name": full_name,
                "full_bio": bio,
                "age": infer_age_from_text(education or bio),
                "hometown": person.get("hometown", "") if person else "",
                "education": education,
                "experience": experience,
                "photo_url": photo_url
            }
            append_csv_row(args.output, csv_fieldnames, row)
        
        # Also save raw profile data as before
        safe = re.sub(r"[^0-9a-zA-Z]+", "_", urlparse(url).netloc + urlparse(url).path)
        raw_path = os.path.join("raw_output", f"{safe}.txt")
        os.makedirs(os.path.dirname(raw_path), exist_ok=True)
        with open(raw_path, "w", encoding="utf-8") as f:
            f.write("PAGE_URL: " + url + "\n\n")
            f.write(site_doc["page_text"] or "")
            f.write("\n\nPROFILES:\n")
            for p in site_doc["profiles"]:
                f.write("-----\n")
                f.write(p["url"] + "\n")
                f.write(p["text"] or "")
                f.write("\n")

    print(f"Done. Results written to CSV: {args.output} and raw text files in output/.")


if __name__ == "__main__":
    main()
