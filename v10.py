import requests
import re
import json
import csv
import os
from datetime import datetime
from groq import Groq

def chunk_text(text, chunk_size=2000):
    """Split text into chunks of words for LLM processing."""
    words = text.split()
    for i in range(0, len(words), chunk_size):
        yield " ".join(words[i:i + chunk_size])

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
    Fetch page content from URL and extract doctors using Groq LLM.
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
        page_text = response.text
    except Exception as e:
        print(f"❌ Failed to fetch {url}: {e}")
        return None

    demo_prompt = (
        "Extract ALL doctors from the text below.\n"
        "Return ONLY a valid JSON array.\n"
        "Each object must include:\n"
        " full_name, full_bio, age, hometown, education, experience, photo_url\n"
        "If missing, use empty string.\n"
    )

    client = Groq(api_key=api_key)
    all_doctors = []

    for chunk in chunk_text(page_text, chunk_size=650):
        prompt = f"{demo_prompt}\nTEXT TO EXTRACT FROM:\n{chunk}"
        try:
            completion = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.6,
                max_completion_tokens=1024,
                top_p=1,
                stream=False
            )

            text = completion.choices[0].message.content

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
                    continue

            if isinstance(parsed, dict):
                parsed = [parsed]

            expected_keys = ["full_name", "full_bio", "age", "hometown", "education", "experience", "photo_url"]
            for entry in parsed:
                if not isinstance(entry, dict):
                    continue
                doctor = {k: str(entry.get(k, "")).strip() for k in expected_keys}
                doctor["website"] = url
                all_doctors.append(doctor)

        except Exception as e:
            print(f"⚠ Groq API extraction failed for chunk: {e}")
            continue

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
    
    fieldnames = ["website", "full_name", "full_bio", "age", "hometown", "education", "experience", "photo_url"]
    
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
                "experience": "",
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
    API_KEY = os.environ.get("groq_api_key")
    INPUT_CSV = "team_page/sample_input.csv"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    OUTPUT_CSV = f"output/result_groq_{timestamp}.csv"
    
    process_urls_and_save_csv(API_KEY, INPUT_CSV, OUTPUT_CSV)

