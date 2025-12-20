# from anyio import sleep
import time
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import csv
import os
from urllib.parse import urlparse
import argparse
from dotenv import load_dotenv
from firecrawl import FirecrawlApp


# Keywords to detect doctor/team pages
KEYWORDS = [
    "meet-the-team", "our-team", "meet-the-doctors", "meet-the-dentists",
    "team",  "staff", "meet", "dentist",
    "providers"
]
timestamp = __import__('datetime').datetime.now().strftime('%Y%m%d_%H%M%S')
os.makedirs('sub_urls', exist_ok=True)


def get_targeted_suburls_firecrawl(input_url, KEYWORDS):
    """
    Finds and filters sub-URLs from a base URL that match specific keywords.
    """
    # print(f"🔍 Mapping {input_url} for keywords: {KEYWORDS}...")
    app = FirecrawlApp(api_key=FIRECRAWL_API_KEY)
    
    try:
        map_result = app.map(url=input_url, limit=10, sitemap="include")
        # print(map_result)
        if not map_result:
            return []
        # print(type(map_result))
        # map_result is a MapData object returned by app.map()
        links_raw = getattr(map_result, "links", [])  # safer than map_result.links
        links_list = list(links_raw)  # convert tuple to list if needed
        urls = []
        for item in links_list:
            # item might be a LinkResult
            if hasattr(item, "url"):
                urls.append(item.url)
            # sometimes Firecrawl returns nested tuples
            elif isinstance(item, tuple):
                for subitem in item:
                    if hasattr(subitem, "url"):
                        urls.append(subitem.url)

        # print("All URLs:", urls)
        # Filter URLs that contain any keyword
        found_urls = [url for url in urls if any(k in url.lower() for k in KEYWORDS)]
        # Sort by keyword priority (earlier keyword in KEYWORDS gets higher priority)
        def keyword_priority(url):
            url_lower = url.lower()
            for index, keyword in enumerate(KEYWORDS):
                if keyword in url_lower:
                    return index
            return len(KEYWORDS)  # if no keyword matched
        found_urls.sort(key=keyword_priority)
        # print("Filtered & prioritized URLs:")
        # for url in found_urls:
        #     print(url)
        # Remove duplicates and return
        return list(dict.fromkeys(found_urls))  # preserve order

    except Exception as e:
        print(f"❌ Error mapping {input_url}: {e}")
        return []

if __name__ == "__main__":
    load_dotenv()
    FIRECRAWL_API_KEY = os.getenv("FIRECRAWL_API_KEY")
    p = argparse.ArgumentParser(description='Find candidate team URLs from a list of websites')
    p.add_argument('-i', '--input', default='input/sample_urls.csv', help='Input CSV path (must not have header)')
    args = p.parse_args()
    with open(args.input, newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        websites = [row[0].strip() for row in reader]

    for website in websites:
        print(f"\n{website}")
        parsed = urlparse(website)
        # Trigger only if there is a path or query beyond "/"
        if (parsed.path and parsed.path != "/") or parsed.query:
            website_new = f"{parsed.scheme}://{parsed.netloc}/"
            print(f"[DEBUG] Adjusted website to {website_new}")
        else:
            website_new = website
        candidate_urls = get_targeted_suburls_firecrawl(website_new, KEYWORDS)

        # print(f"Candidate team URLs:")
        for u in candidate_urls:
            print("-", u)

        with open(f'sub_urls/url_candidates_{timestamp}.csv', 'a', encoding='utf-8') as fh:
            if candidate_urls:
                # Join all URLs for this website with commas
                line = ",".join([website] + candidate_urls)
                fh.write(line + "\n")
            else:
                fh.write(website + ",\n")  # No candidates found
        time.sleep(10)