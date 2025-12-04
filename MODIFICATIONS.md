# Script Modifications - Llama-3.2-3B Integration

## Summary
Modified `team_page_scraper.py` to use **Llama-3.2-3B** instead of heuristic extraction. The script now follows a pure LLM-based approach with no pattern matching.

## Key Changes

### 1. **Pipeline Steps**
The script now implements the exact 5-step process you requested:

- ✅ **Step 1**: Extract HTML text from each URL
- ✅ **Step 2**: Clean text (remove scripts, styles, comments, language keywords)
- ✅ **Step 3**: Save raw cleaned text to `raw_output/` folder (named same as website)
- ✅ **Step 4**: Pass cleaned text to Llama-3.2-3B and get response
- ✅ **Step 5**: Save/append response to `output/results_{timestamp}.csv`

### 2. **Removed Components**
- ❌ All heuristic extraction logic (`find_profile_links()`, pattern matching)
- ❌ Profile link following (no sub-page crawling)
- ❌ `heuristic_extract_people()` fallback function
- ❌ `infer_age_from_text()` (LLM extracts this directly)
- ❌ `visible_text_from_soup()` (replaced with `clean_text_content()`)

### 3. **New/Modified Functions**

#### `clean_text_content(html_text)` - NEW
- Removes scripts, styles, comments, meta tags
- Removes language keywords (JavaScript, CSS, HTML, etc.)
- Collapses extra whitespace
- Returns clean, readable text suitable for LLM

#### `extract_site_documents(url)` - MODIFIED
- Now only extracts and cleans the main page
- No profile link following
- Returns cleaned text directly
- Much simpler and faster than before

#### `init_llm(model_name)` - UPDATED
- Uses `meta-llama/Llama-3.2-3B` by default
- Uses `text-generation` pipeline for causal LM
- Configures device mapping for GPU/CPU auto-detection
- Temperature set to 0.0 for deterministic output

#### `ask_llm_for_extraction(pipe, cleaned_text, website_url)` - NEW (renamed from ask_llm_for_json)
- Passes cleaned text directly to Llama
- Requests CSV format extraction
- Parses CSV output from LLM
- Saves raw LLM responses for debugging

#### `main()` - SIMPLIFIED
- Removed `--save-raw-only` flag (always saves raw text now)
- Removed model fallback logic
- Required LLM initialization with proper error handling
- Clear progress output with emojis for readability
- Saves both raw cleaned texts AND CSV results

### 4. **Output Structure**

```
raw_output/
├── 1500dental_com_our_team.txt          # Site 1 cleaned text
├── 18thstreetdental_com_index_php_...txt # Site 2 cleaned text
└── ...

output/
├── result_20251204_120000.csv           # Final CSV with all extracted data
└── llm_raw/
    ├── 1500dental_com_our_team_response.txt   # Raw LLM output for debugging
    └── ...
```

### 5. **CSV Output Format**
Same fields as before:
- `website` - Source URL
- `full_name` - Person's full name
- `full_bio` - Brief bio/description
- `age` - Age (extracted by LLM from text context)
- `hometown` - Hometown/location (extracted by LLM)
- `education` - Education details (extracted by LLM)
- `experience` - Experience/credentials (extracted by LLM)
- `photo_url` - Photo URL if available (extracted by LLM)

## Usage

### Installation
```bash
pip install -r team_requirements.txt
```

### Basic Usage
```bash
python team_page_scraper.py --input team_page/sample_input.csv --output output/result_{timestamp}.csv
```

### With Custom Model
```bash
python team_page_scraper.py \
    --input team_page/input1.csv \
    --output output/custom_result.csv \
    --model meta-llama/Llama-3.2-3B
```

## Requirements

### System Requirements
- **RAM**: 16GB+ (Llama-3.2-3B is 3B parameters)
- **GPU**: Recommended 8GB+ VRAM (can use CPU but will be slow)
- **Storage**: ~7GB for model download

### Python Packages (Updated)
- `transformers>=4.40.0` - For LLM pipelines
- `torch>=2.0.0` - ML framework
- `accelerate>=0.24.0` - Device mapping for GPU/CPU
- `bitsandbytes>=0.41.0` - 8-bit quantization support (optional but recommended)
- `beautifulsoup4>=4.12.2` - HTML parsing
- `requests>=2.28` - URL fetching

## Model Access

Note: To use `meta-llama/Llama-3.2-3B`, you need:
1. Hugging Face account (free)
2. Accept Meta's license: https://huggingface.co/meta-llama/Llama-3.2-3B
3. Create a token at https://huggingface.co/settings/tokens
4. Authenticate: `huggingface-cli login`

## Performance Notes

- **Model Size**: 3B parameters (smaller, faster than larger Llamas)
- **Speed**: ~10-30 seconds per page depending on text length and hardware
- **VRAM Usage**: ~6-8GB typical
- **Accuracy**: Pure LLM-based, more consistent than heuristics

## Debugging

Raw LLM responses are saved in `output/llm_raw/` for each URL. Check these files to see:
- Exact CSV output from Llama
- Extraction quality
- Any parsing issues

If LLM output format differs, adjust the CSV parsing logic in `ask_llm_for_extraction()`.

## Migration from Old Version

Changes to existing code:
- Remove any calls to `heuristic_extract_people()`
- Remove `--save-raw-only` flag usage
- Remove `--model google/flan-t5-small` (now requires GPU for local LLM)
- Update any external dependencies on removed functions
