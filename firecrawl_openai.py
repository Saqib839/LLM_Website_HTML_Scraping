import json
import os
import time
from openai import OpenAI
from firecrawl import FirecrawlApp
from dotenv import load_dotenv

load_dotenv()

# API Keys
OPENAI_CLIENT = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
FIRECRAWL_APP = FirecrawlApp(api_key=os.getenv("FIRECRAWL_API_KEY"))

# Configuration
INPUT_URLS = ["https://18thstreetdental.com/", "https://example-dentist.com/"] # Load your 30k here
BATCH_INPUT_FILE = "openai_batch_tasks.jsonl"
OUTPUT_DATA_FILE = "final_doctors_data.jsonl"

def phase_1_bulk_scrape(urls):
    """Uses Firecrawl to turn 30k URLs into clean Markdown text."""
    print(f"--- Phase 1: Scraping {len(urls)} URLs ---")
    # Note: For 30k URLs, Firecrawl handles this in the background
    # This example shows the logic for starting the job
    batch_job = FIRECRAWL_APP.start_batch_scrape(
        urls=urls,
        params={'formats': ['markdown'], 'onlyMainContent': True}
    )
    print(f"Batch Scrape started. Job ID: {batch_job['id']}")
    
    # In a real 30k run, you'd wait a few hours and then collect results
    return batch_job['id']

def phase_2_prepare_and_submit_to_openai(scraped_data_list):
    """Formats scraped text into OpenAI's Batch JSONL format and submits."""
    print("--- Phase 2: Submitting to OpenAI Batch API (50% Discount) ---")
    
    with open(BATCH_INPUT_FILE, "w") as f:
        for idx, item in enumerate(scraped_data_list):
            url = item.get('url')
            content = item.get('markdown', '')[:15000] # Cap content to save tokens
            
            # Each line is a separate 'request' object
            task = {
                "custom_id": f"request_{idx}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": "gpt-4o-mini",
                    "messages": [
                        {"role": "system", "content": "Extract doctor info: full_name, bio, education, designation (Owner/Associate). Return ONLY JSON."},
                        {"role": "user", "content": f"URL: {url}\n\nContent: {content}"}
                    ],
                    "response_format": { "type": "json_object" }
                }
            }
            f.write(json.dumps(task) + "\n")

    # Upload file to OpenAI
    batch_file = OPENAI_CLIENT.files.create(file=open(BATCH_INPUT_FILE, "rb"), purpose="batch")
    
    # Start the Batch Job
    openai_batch = OPENAI_CLIENT.batches.create(
        input_file_id=batch_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h"
    )
    print(f"OpenAI Batch Job Created: {openai_batch.id}")
    return openai_batch.id

def phase_3_check_and_download(batch_id):
    """Checks if OpenAI is done (usually takes 1-24 hours)."""
    status = OPENAI_CLIENT.batches.retrieve(batch_id)
    print(f"Current Status: {status.status}")
    
    if status.status == "completed":
        file_response = OPENAI_CLIENT.files.content(status.output_file_id)
        with open(OUTPUT_DATA_FILE, "w") as f:
            f.write(file_response.text)
        print(f"Done! Results saved to {OUTPUT_DATA_FILE}")
    else:
        print("Batch still processing. Try again in 1 hour.")

# --- EXECUTION FLOW ---
# 1. Start Firecrawl (Wait for it to finish)
# 2. results = FIRECRAWL_APP.get_batch_scrape_status(job_id)
# 3. phase_2_prepare_and_submit_to_openai(results['data'])
# 4. phase_3_check_and_download(openai_batch_id)

import json
import os
import pandas as pd
from openai import OpenAI
from firecrawl import FirecrawlApp
from dotenv import load_dotenv

load_dotenv()

# --- INITIALIZATION ---
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
app = FirecrawlApp(api_key=os.getenv("FIRECRAWL_API_KEY"))

# Files
INPUT_CSV = "your_30k_list.csv"  # Ensure this has a column named 'url'
SCRAPED_DATA_JSON = "scraped_results.json"
BATCH_INPUT_FILE = "to_openai_batch.jsonl"

def load_urls(file_path):
    """Load URLs from CSV using Pandas for speed."""
    df = pd.read_csv(file_path)
    return df['url'].tolist()

def run_firecrawl_batch(urls):
    """Phase 1: Scrape everything into Markdown."""
    print(f"🚀 Starting Firecrawl Batch for {len(urls)} URLs...")
    # Firecrawl handles batching internally; we start the job and get an ID
    job = app.start_batch_scrape(urls, {
        'formats': ['markdown'],
        'onlyMainContent': True
    })
    return job['id']

def poll_firecrawl(job_id):
    """Wait for Firecrawl to finish (30k sites may take 1-3 hours)."""
    while True:
        status = app.get_batch_scrape_status(job_id)
        print(f"Firecrawl Status: {status['status']} ({status['completed']}/{status['total']})")
        if status['status'] == 'completed':
            return status['data']
        time.sleep(60) # Check every minute

def create_openai_batch_file(scraped_data):
    """Phase 2: Prepare the 50% discount Batch file."""
    print("📝 Preparing OpenAI Batch file...")
    with open(BATCH_INPUT_FILE, "w") as f:
        for idx, item in enumerate(scraped_data):
            url = item.get('metadata', {}).get('sourceURL', 'unknown')
            markdown = item.get('markdown', '')[:10000] # Cap text to avoid bloat
            
            # OpenAI Batch Request Format
            request = {
                "custom_id": f"site_{idx}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": "gpt-4o-mini",
                    "messages": [
                        {"role": "system", "content": "Extract doctors into JSON: name, bio, designation (Owner/Associate)."},
                        {"role": "user", "content": f"Source: {url}\n\nContent: {markdown}"}
                    ],
                    "response_format": { "type": "json_object" }
                }
            }
            f.write(json.dumps(request) + "\n")

def submit_to_openai():
    """Phase 3: Upload and start OpenAI Batch."""
    file_upload = client.files.create(file=open(BATCH_INPUT_FILE, "rb"), purpose="batch")
    batch_job = client.batches.create(
        input_file_id=file_upload.id,
        endpoint="/v1/chat/completions",
        completion_window="24h"
    )
    print(f"✅ OpenAI Batch Started! ID: {batch_job.id}")
    return batch_job.id

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    urls = load_urls(INPUT_CSV)
    
    # Run Scrape
    fire_id = run_firecrawl_batch(urls)
    results = poll_firecrawl(fire_id)
    
    # Save results locally as backup
    with open(SCRAPED_DATA_JSON, "w") as f:
        json.dump(results, f)
        
    # Send to OpenAI
    create_openai_batch_file(results)
    submit_to_openai()