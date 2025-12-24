import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import csv
import os
from urllib.parse import urlparse
import argparse
import re
import json
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv

# Keywords to detect doctor/team pages
KEYWORDS = [
    "meet-the-team", "our-team", "meet-the-doctors", "meet-the-dentists",
    "team",  "staff", "meet", "providers", 'site-map', 'sitemap'
]
timestamp = __import__('datetime').datetime.now().strftime('%Y%m%d_%H%M%S')
os.makedirs('sub_urls', exist_ok=True)

def get_sitemap_urls(base_url):
    """Try to fetch sitemap.xml and return all URLs inside it."""
    sitemap_url = urljoin(base_url, "/sitemap.xml")
    try:
        resp = requests.get(sitemap_url, timeout=10)
        if resp.status_code != 200:
            return []
        soup = BeautifulSoup(resp.text, "xml")
        urls = [loc.text for loc in soup.find_all("loc")]
        return urls
    except Exception:
        return []

def get_homepage_urls(base_url):
    """Fetch homepage and return all internal links."""
    try:
        resp = requests.get(base_url, timeout=10)
        if resp.status_code != 200:
            return []
        soup = BeautifulSoup(resp.text, "html.parser")
        links = set()
        for tag in soup.find_all("a", href=True):
            href = tag['href']
            # Make absolute URL
            full_url = urljoin(base_url, href)
            # Keep only internal links
            if urlparse(full_url).netloc == urlparse(base_url).netloc:
                links.add(full_url)
        return list(links)
    except Exception:
        return []

def find_candidate_doctor_pages(base_url, max_results=5):
    """
    Given a single website URL, return top candidate URLs
    that might contain doctor/team information.
    """
    # Step 1: Try sitemap
    urls = get_sitemap_urls(base_url)
    
    # Step 2: Fallback to homepage links
    if not urls:
        urls = get_homepage_urls(base_url)

    # Step 3: Filter URLs by keywords
    candidates = [u for u in urls if any(k in u.lower() for k in KEYWORDS)]

    # Step 4: Remove duplicates and limit results
    candidates = list(dict.fromkeys(candidates))  # preserve order
    return candidates[:max_results]

def url_exists(url):
    HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/142.0.0.0 Safari/537.36"
    }
    try:
        r = requests.get(url, timeout=10, allow_redirects=True, headers=HEADERS)
        return r.status_code == 200
    except Exception as e:
        print(f"[DEBUG] url_exists exception for {url}: {e}")
        return False


def manual_test(base_url):
    print(f"[DEBUG] Starting manual test")

    COMMON_TEAM_PATHS = [
    "/about-us",
    "/about",
    "/meet-us"
    #"/our-team",
    #"/meet-the-doctor",
    #"/meet-the-doctors", 
    #"/about-us/team",
    #"/about/team",
    #"/our-staff",
    #"/staff",
    #"/providers",
    ]
    found = []
    for path in COMMON_TEAM_PATHS:
        full_url = urljoin(base_url, path)
        if url_exists(full_url):
            # print(f"[DEBUG] manual test found valid URL: {full_url}")
            found.append(full_url)
    return found

def url_priority(url):
    url = url.lower()
    for idx, keyword in enumerate(KEYWORDS):
        if keyword in url:
            # Earlier keyword = higher priority
            return len(KEYWORDS) - idx
    return 0
    
    
def chunk_text(text, chunk_size=20000):
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

def extract_visible_text(html_content):
    # Parse HTML
    soup = BeautifulSoup(html_content, "html.parser")
    
    # Remove all scripts, styles, head, meta, and comments
    for tag in soup(["script", "style", "head", "meta", "link", "noscript"]):
        tag.decompose()
    
    # Get visible text
    text = soup.get_text(separator="\n")
    
    # Clean multiple blank lines
    lines = [line.strip() for line in text.splitlines()]
    visible_text = "\n".join([line for line in lines if line])
    
    return visible_text

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
        html_text = response.text
        # print(html_text)


        # os.makedirs("raw_html", exist_ok=True)
        # # Make a safe filename from URL
        # safe_filename = re.sub(r'[^a-zA-Z0-9_-]', '_', url)
        # file_path = f"raw_html/{safe_filename}.html"
        # with open(file_path, "w", encoding="utf-8") as debug_file:
        #     debug_file.write(html_text)
        # print(f"✓ Saved raw HTML for {url} → {file_path}")

        page_text = extract_visible_text(html_text)
    except Exception as e:
        print(f"❌ Failed to fetch {url}: {e}")
        return None

    demo_prompt = (
        "Extract ALL doctors from the text below.\n"
        "Return ONLY a valid JSON array.\n"
        "Each object must include:\n"
        "  full_name, full_bio, age, hometown, education, graduation_year, designation, photo_url\n"
        "Rules:\n"
        "  - If any of field is missing, use empty string.\n"
        "  - Determine designation as 'Owner' or 'Associate' using the weighted score below:\n"
        "      a) Name prominence across the text.\n"
        "      b) Listing order (earlier listed gets higher weight).\n"
        "      Highest score → Owner, others → Associate.\n"
    )

    # Configure OpenAI key for this request
    all_doctors = []
    client = OpenAI(api_key=api_key)

    if len(page_text) < 30000:
        for chunk in chunk_text(page_text, chunk_size=20000):
            print(f"Processing chunk of size {len(chunk)}")
            prompt = f"{demo_prompt}\nTEXT TO EXTRACT FROM:\n{chunk}"
            try:
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}]
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
                        continue

                if isinstance(parsed, dict):
                    parsed = [parsed]

                expected_keys = ["full_name", "full_bio", "age", "hometown", "education", "graduation_year", "designation", "photo_url"]
                for entry in parsed:
                    if not isinstance(entry, dict):
                        continue
                    doctor = {k: str(entry.get(k, "")).strip() for k in expected_keys}
                    doctor["website"] = url
                    all_doctors.append(doctor)

            except Exception as e:
                print(f"⚠ OpenAI API extraction failed for chunk: {e}")
                continue
    else:
        print(f"Too long page text {len(page_text)}, skipping.")
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
    
    # print(f"✓ Appended {len(rows)} row(s) to {csv_path}")

def process_urls_and_save_csv(api_key, url, output_csv):
    """
    Process each URL from input CSV one by one and append results to output CSV.
    """
    seen_people = {}  # Track (website, name) to avoid duplicates
    
    if url:
        # print(f"\n{'='*80}")
        # print(f"Processing: {url}")
        # print('='*80)
        
        # Extract doctors from URL
        doctors = extract_doctors_from_url(api_key, url)
        
        # print(f"✓ Extracted {len(doctors)} doctor(s)")
        
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



if __name__ == "__main__":
    p = argparse.ArgumentParser(description='Find candidate team URLs from a list of websites')
    p.add_argument('-i', '--input', default='websites_b1.csv', help='Input CSV path (must not have header)')
    args = p.parse_args()
    fieldnames = ["website", "full_name", "full_bio", "age", "hometown", "education", "graduation_year", "designation", "photo_url"]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    OUTPUT_CSV = f"output/result_openai_{timestamp}.csv"
    write_csv_header(OUTPUT_CSV, fieldnames)

    with open(args.input, newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        websites = [row[0].strip() for row in reader]

    for website in websites:
        print(f"\n{website}")
        parsed = urlparse(website)
        # Trigger only if there is a path or query beyond "/"
        if (parsed.path and parsed.path != "/") or parsed.query:
            website_new = f"{parsed.scheme}://{parsed.netloc}/"
            #print(f"[DEBUG] Adjusted website to {website_new}")
        else:
            website_new = website
        
        # candidate_urls = find_candidate_doctor_pages(website_new)
        candidate_urls = []
        if len(candidate_urls) == 0:
            #print(f"[INFO] No candidate URLs found")
            candidate_urls = manual_test(website_new)

        # Sort candidate URLs for consistency like meet-the-team first then our-team
        candidate_urls.sort(key=url_priority, reverse=True)

        # print(f"Candidate team URLs:")
        for u in candidate_urls:
            print("-", u)
            load_dotenv()
            API_KEY = os.getenv("OPENAI_API_KEY")
            if not API_KEY:
                print("❌ OPENAI_API_KEY not set. Please add it to your environment or .env file.")
                exit(1)
            process_urls_and_save_csv(API_KEY, u, OUTPUT_CSV)

