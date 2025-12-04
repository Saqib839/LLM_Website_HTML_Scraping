# Pipeline Flow Diagram

## Five-Step Process

```
┌─────────────────────────────────────────────────────────────────────┐
│                         INPUT: URLs from CSV                         │
│                    team_page/input1.csv (one URL/line)              │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
        ┌────────────────────────────────────────┐
        │  STEP 1: Extract HTML Text             │
        │  ─────────────────────────────────────  │
        │  • Fetch URL with requests              │
        │  • User-agent rotation for 403 bypass   │
        │  • Returns: raw HTML content            │
        └────────────────────┬───────────────────┘
                             │
                             ▼
        ┌────────────────────────────────────────┐
        │  STEP 2: Clean Text                    │
        │  ─────────────────────────────────────  │
        │  • Parse HTML with BeautifulSoup        │
        │  • Remove scripts, styles, comments     │
        │  • Remove meta, link, noscript tags     │
        │  • Remove JS keywords (var, function)   │
        │  • Collapse extra whitespace            │
        │  • Returns: clean text string           │
        └────────────────────┬───────────────────┘
                             │
                             ▼
        ┌────────────────────────────────────────┐
        │  STEP 3: Save Raw Text to File          │
        │  ─────────────────────────────────────  │
        │  • Output directory: raw_output/        │
        │  • Filename: {website_domain_path}.txt  │
        │  • Content: cleaned text (no HTML)      │
        │  • Format: human-readable               │
        └────────────────────┬───────────────────┘
                             │
                             ▼
        ┌────────────────────────────────────────┐
        │  STEP 4: Query Llama-3.2-3B            │
        │  ─────────────────────────────────────  │
        │  • Initialize model (first run = 7GB)   │
        │  • Create extraction prompt              │
        │  • Pass cleaned text to LLM             │
        │  • Generate CSV response                │
        │  • Parse CSV output                     │
        │  • Returns: List of dicts per person    │
        └────────────────────┬───────────────────┘
                             │
                             ▼
        ┌────────────────────────────────────────┐
        │  STEP 5: Append to Results CSV          │
        │  ─────────────────────────────────────  │
        │  • Output: output/result_{timestamp}.csv│
        │  • Columns: website, full_name,         │
        │             full_bio, age, hometown,    │
        │             education, experience,      │
        │             photo_url                   │
        │  • De-duplication: skip seen people     │
        └────────────────────┬───────────────────┘
                             │
                    ┌────────┴────────┐
                    │                 │
                    ▼                 ▼
         ┌──────────────────┐  ┌──────────────────┐
         │ OUTPUTS (CSV)    │  │ DEBUG (Raw LLM)  │
         │ ──────────────── │  │ ────────────────  │
         │ result_20251204_ │  │ output/llm_raw/  │
         │ 120500.csv       │  │ {site}_response  │
         └──────────────────┘  └──────────────────┘
```

## Data Flow Example

### Input
```
http://1500dental.com/our-team
```

### Step 1-2: HTML → Cleaned Text
```html
<!-- Raw HTML -->
<script>var x = 1;</script>
<style>body { color: black; }</style>
<h1>Meet Our Team</h1>
<p>Dr. John Smith is our lead dentist.</p>

↓ BECOMES ↓

Meet Our Team Dr. John Smith is our lead dentist.
```

### Step 3: Saved to File
```
File: raw_output/1500dental_com_our_team.txt
─────────────────────────────────────────────
Website: http://1500dental.com/our-team
Cleaned Text:
================================================================================
Meet Our Team Dr. John Smith is our lead dentist with 20 years of experience...
```

### Step 4: LLM Extraction
```
Input Prompt to Llama-3.2-3B:
─────────────────────────────
You are an information extraction assistant...
Return ONLY CSV rows with columns:
website,full_name,full_bio,age,hometown,education,experience,photo_url

TEXT TO EXTRACT FROM:
Meet Our Team Dr. John Smith is our lead dentist with 20 years of experience...

LLM Response:
─────────────
http://1500dental.com/our-team,John Smith,Lead dentist,45,Chicago IL,DDS NYU 2000,20 years private practice,
```

### Step 5: Results CSV
```csv
website,full_name,full_bio,age,hometown,education,experience,photo_url
http://1500dental.com/our-team,John Smith,Lead dentist,45,Chicago IL,DDS NYU 2000,20 years private practice,
...
```

## Processing Time Estimates

```
Per URL:
├─ Step 1 (Fetch): 2-5 seconds
├─ Step 2 (Clean): <100ms
├─ Step 3 (Save):  <10ms
├─ Step 4 (LLM):   10-30 seconds (GPU) or 60-120s (CPU)
└─ Step 5 (CSV):   <10ms
────────────────────────
Total per URL: ~15-35 seconds (GPU) or ~70-125 seconds (CPU)
```

For 100 URLs:
- GPU: ~25-60 minutes
- CPU: ~2-3 hours

## Error Handling

```
During execution:
├─ URL fetch fails → Write empty CSV row → Continue
├─ HTML parsing fails → Write empty CSV row → Continue
├─ LLM error → Log error → Write empty CSV row → Continue
├─ CSV parsing fails → Log warning → Skip entry
└─ Duplicate person → Skip (de-duplication)

No stops, just continues to next URL with logging.
```

## Files Created/Modified

```
Before:                           After:
─────────────────────            ─────────────────────
(no raw_output)          ──→     raw_output/
                                 ├── site1.txt
                                 ├── site2.txt
                                 └── ...

output/
├── llm_raw/              (NEW)
│   ├── site1_response.txt
│   └── ...
└── result_20251204_*.csv
```

## Exit Conditions

Script completes successfully when:
✅ All URLs processed (even if some failed)
✅ CSV file written with all results
✅ Raw text files saved
✅ LLM debug outputs saved
✅ Deduplication applied

Script exits with error only if:
❌ Can't initialize Llama-3.2-3B model
❌ Can't read input CSV
❌ Critical I/O error
