import requests
# import re
import json
import csv
import os
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv
# from bs4 import BeautifulSoup

def read_urls_from_csv(csv_path):
    """Read URLs from CSV file (one per line or comma-separated)."""
    urls = []
    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # Support both single URLs and comma-separated URLs
                if "," in line and not line.startswith("http"):
                    urls.extend([url.strip() for url in line.split(",") if url.strip()])
                else:
                    urls.append(line)
        print(f"✓ Loaded {len(urls)} URLs from {csv_path}")
    except Exception as e:
        print(f"❌ Failed to read URLs from {csv_path}: {e}")
    return urls

def extract_doctors_from_url(api_key, url):
    """
    Fetch page content from URL and extract doctors using OpenAI LLM.
    Handles token limits by chunking the text.
    """
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                          "AppleWebKit/537.36 (KHTML, like Gecko) "
                          "Chrome/142.0.0.0 Safari/537.36"
        }
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()

    except Exception as e:
        print(f"❌ Failed to fetch {url}: {e}")
        return None

    demo_prompt = (
        "Extract ALL doctors from the website at the URL below.\n"
        "Return ONLY a valid JSON array.\n"
        "Each object must include:\n"
        "  full_name, full_bio, age, hometown, education, designation, photo_url\n"
        "Rules:\n"
        "  - If graduation year is mentioned in bio, calculate age as 26 + (current_year - graduation_year). Otherwise, use empty string.\n"
        "  - If hometown, education, or photo_url is missing, use empty string.\n"
        "  - Determine designation as 'Owner' or 'Associate' using the following logic:\n"
        "      1. If only one doctor → Owner.\n"
        "      2. If practice name contains a doctor's last name → Owner.\n"
        "      3. Otherwise, compute weighted score based on:\n"
        "         a) Name prominence across the website.\n"
        "         b) Listing order (earlier listed gets higher weight).\n"
        "         c) Likely owner age range (35-55 years).\n"
        "         Highest score → Owner, others → Associate.\n"
    )

    all_doctors = []
    prompt = f"{demo_prompt}\nURL TO EXTRACT FROM:\n{url}"

    # Configure OpenAI key for this request
    client = OpenAI(api_key=api_key)
    try:
        response = client.responses.create(
            model="o4-mini-deep-research",
            tools=[{"type": "web_search"}],
            input= prompt
            )

        text = response.choices[0].message.content

        if text.startswith(prompt):
            text = text[len(prompt):]

        def _extract_json(s):
            if not s:
                return None
            s = s.strip()
            for start_char, end_char in [('[', ']'), ('{', '}')]:
                start = s.find(start_char)
                end = s.rfind(end_char)
                if start != -1 and end != -1 and end > start:
                    return s[start:end+1]
            return None

        json_sub = _extract_json(text)
        parsed = None
        if json_sub:
            try:
                parsed = json.loads(json_sub)
            except json.JSONDecodeError:
                pass

        if parsed is None:
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                print(f"⚠ JSON decode error for {url}.")
                preview = (text[:1000] + '...') if text and len(text) > 1000 else text
                print(f"Preview of response:\n{preview}")
                return None

        if isinstance(parsed, dict):
            parsed = [parsed]

        expected_keys = ["full_name", "full_bio", "age", "hometown", "education", "designation", "photo_url"]
        for entry in parsed:
            if not isinstance(entry, dict):
                continue
            doctor = {k: str(entry.get(k, "")).strip() for k in expected_keys}
            doctor["website"] = url
            all_doctors.append(doctor)

    except Exception as e:
        print(f"⚠ OpenAI API extraction failed for {url}: {e}")
        return None

    return all_doctors if all_doctors else None

def write_csv_header(csv_path, fieldnames):
    """Write CSV header (creates new file)."""
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
    print(f"✓ Created CSV header in {csv_path}")

def append_to_csv(csv_path, fieldnames, rows):
    """Append rows to CSV file (or create if not exists)."""
    if not rows:
        return
    
    # Check if file exists
    file_exists = os.path.exists(csv_path)
    
    with open(csv_path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerows(rows)
    
    print(f"✓ Appended {len(rows)} row(s) to {csv_path}")

def process_urls_and_save_csv(api_key, input_csv, output_csv):
    """
    Process each URL from input CSV one by one and append results to output CSV.
    """
    urls = read_urls_from_csv(input_csv)
    if not urls:
        print("❌ No URLs found!")
        return
    
    fieldnames = ["website", "full_name", "full_bio", "age", "hometown", "education", "designation", "photo_url"]
    
    # Initialize output CSV
    write_csv_header(output_csv, fieldnames)
    
    seen_people = {}  # Track (website, name) to avoid duplicates
    
    for idx, url in enumerate(urls, 1):
        print(f"\n{'='*80}")
        print(f"[{idx}/{len(urls)}] Processing: {url}")
        print('='*80)
        
        # Extract doctors from URL
        doctors = extract_doctors_from_url(api_key, url)

        if not doctors:
            print(f"⚠ No doctors extracted from {url}")
            # Still append an empty row to track this URL
            append_to_csv(output_csv, fieldnames, [{
                "website": url,
                "full_name": "",
                "full_bio": "",
                "age": "",
                "hometown": "",
                "education": "",
                "designation": "",
                "photo_url": ""
            }])
            continue
        
        print(f"✓ Extracted {len(doctors)} doctor(s)")
        
        # Filter duplicates and prepare rows for CSV
        csv_rows = []
        for doctor in doctors:
            person_key = (url, doctor.get("full_name", "").strip())
            if person_key not in seen_people:
                seen_people[person_key] = True
                csv_rows.append(doctor)
        
        # Append to CSV
        if csv_rows:
            append_to_csv(output_csv, fieldnames, csv_rows)
        else:
            print(f"⚠ All doctors from {url} were duplicates, skipping")
    
    print(f"\n{'='*80}")
    print(f"✓ All URLs processed! Results saved to: {output_csv}")
    print('='*80)


if __name__ == "__main__":
    load_dotenv()
    API_KEY = os.getenv("OPENAI_API_KEY")
    if not API_KEY:
        print("❌ OPENAI_API_KEY not set. Please add it to your environment or .env file.")
        exit(1)
    INPUT_CSV = "team_page/sample_input.csv"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    OUTPUT_CSV = f"output/result_openai_{timestamp}.csv"
    
    process_urls_and_save_csv(API_KEY, INPUT_CSV, OUTPUT_CSV)

