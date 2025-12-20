from firecrawl import FirecrawlApp
from dotenv import load_dotenv
import os
load_dotenv()
API_KEY = os.getenv("FIRECRAWL_API_KEY")

app = FirecrawlApp(api_key=API_KEY)

def get_targeted_suburls(input_url, KEYWORDS):
    """
    Finds and filters sub-URLs from a base URL that match specific keywords.
    """
    print(f"🔍 Mapping {input_url} for keywords: {KEYWORDS}...")
    
    try:
        map_result = app.map(url=input_url, limit=50, sitemap="include")
        # print(map_result)
        if not map_result or 'links' not in map_result:
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
        print("Filtered & prioritized URLs:")
        for url in found_urls:
            print(url)
        # Remove duplicates and return
        # return list(set(found_urls))

    except Exception as e:
        print(f"❌ Error mapping {input_url}: {e}")
        return []

# --- Example Usage ---
KEYWORDS = [
    "meet-the-team", "our-team",  "meet-the-dentists", "meet-the-doctors",
    "team",  "staff", "meet", "providers"
]
target_site = "http://redwingdentalcare.com/"

matching_links = get_targeted_suburls(target_site, KEYWORDS)

# print(f"\n✅ Found {len(matching_links)} relevant sub-URLs:")
# for link in matching_links:
#     print(f" - {link}")