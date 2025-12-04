# Quick Reference

## Installation

```bash
# 1. Install dependencies
pip install -r team_requirements.txt

# 2. First time: Login to Hugging Face for model access
huggingface-cli login
# Then accept Meta's Llama license at:
# https://huggingface.co/meta-llama/Llama-3.2-3B
```

## Usage

### Basic Run
```bash
python team_page_scraper.py --input team_page/input1.csv
# Output: output/result_YYYYMMDD_HHMMSS.csv
```

### Custom Output File
```bash
python team_page_scraper.py \
    --input team_page/input1.csv \
    --output output/my_results.csv
```

### Use Different Model
```bash
# Use a different Llama version
python team_page_scraper.py \
    --input team_page/input1.csv \
    --model meta-llama/Llama-3.2-1B  # Smaller, faster
    
# Or any HuggingFace model
python team_page_scraper.py \
    --input team_page/input1.csv \
    --model meta-llama/Llama-3.2-11B  # Larger, more accurate
```

## Input Format

`team_page/input1.csv` - One URL per line, no header:
```
http://1500dental.com/our-team
http://18thstreetdental.com/index.php/meet-the-doctor
https://1westparkdental.com/dental-insurance-providers
...
```

## Output Files

After running, you'll have:

```
raw_output/
├── 1500dental_com_our_team.txt           # Cleaned text from URL 1
├── 18thstreetdental_com_index_php_meet_the_doctor.txt  # URL 2
└── ...

output/
├── result_20251204_120500.csv            # MAIN RESULTS
└── llm_raw/                               # For debugging
    ├── 1500dental_com_our_team_response.txt
    ├── 18thstreetdental_com_index_php_meet_the_doctor_response.txt
    └── ...
```

## CSV Output Format

**Filename**: `result_YYYYMMDD_HHMMSS.csv`

**Columns**:
- `website` - Source URL
- `full_name` - Person's name
- `full_bio` - Biography/description
- `age` - Age (extracted by LLM)
- `hometown` - City/location
- `education` - Degree/school info
- `experience` - Years/credentials
- `photo_url` - Photo URL if available

**Example row**:
```csv
http://1500dental.com/our-team,Dr. John Smith,"20+ years dental experience",45,"Chicago, IL","DDS from Northwestern University 1999","20 years private practice, Board Certified",
```

## Troubleshooting

### Error: "Failed to initialize LLM"
```
Solution:
1. Check internet connection (needs to download model)
2. Run: huggingface-cli login
3. Accept Meta's license: https://huggingface.co/meta-llama/Llama-3.2-3B
4. Try again
```

### Error: "CUDA out of memory"
```
Solution:
1. Use smaller model:
   python team_page_scraper.py --model meta-llama/Llama-3.2-1B
   
2. Or use CPU (slower but works):
   # Edit init_llm() to remove device_map settings
```

### Error: "No module named 'transformers'"
```
Solution:
pip install --upgrade transformers torch
```

### Error: "Could not fetch URL"
```
Solution:
1. Check if URL is valid and accessible
2. Website might block scrapers
3. Check output/llm_raw/ for any partial responses
4. Script continues despite individual URL failures
```

### LLM returning empty results
```
Check:
1. output/llm_raw/{site}_response.txt for what LLM returned
2. raw_output/{site}.txt to see what text was sent
3. Try with a different model that's larger
4. Check your demo_prompt.txt if you created one
```

## Performance Tips

### Faster Processing
```bash
# Use smaller model (1B instead of 3B)
python team_page_scraper.py --model meta-llama/Llama-3.2-1B

# Expected: ~5 seconds/URL on GPU instead of 15-30
```

### Better Accuracy
```bash
# Use larger model (11B instead of 3B)
# WARNING: Requires 16GB+ VRAM
python team_page_scraper.py --model meta-llama/Llama-3.2-11B
```

### Process Multiple Files
```bash
# Run 3 separate instances
python team_page_scraper.py --input input1.csv --output output/result1.csv
python team_page_scraper.py --input input2.csv --output output/result2.csv
python team_page_scraper.py --input input3.csv --output output/result3.csv

# Combine results
cat output/result*.csv > output/combined.csv
```

## Monitoring Progress

The script shows:
- Processing bar with URL
- Text extraction size
- LLM query status
- Rows written count
- Raw file locations

**Example output**:
```
================================================================================
Processing: http://1500dental.com/our-team
================================================================================
✓ Extracted text: 5243 characters
✓ Saved raw cleaned text to: raw_output/1500dental_com_our_team.txt
🔄 Querying Llama-3.2-3B for extraction...
✓ Extracted 3 person/people
✓ Wrote 3 row(s) to CSV
```

## Debugging

### Check what text LLM received
```bash
cat raw_output/{site_name}.txt
```

### Check what LLM returned
```bash
cat output/llm_raw/{site_name}_response.txt
```

### Test with single URL
```bash
# Create temp file with one URL
echo "http://example.com/team" > test_input.csv

# Run
python team_page_scraper.py --input test_input.csv --output output/test.csv

# Check results
cat output/test.csv
cat output/llm_raw/example_com_team_response.txt
```

## Model Options

| Model | Size | Speed | VRAM | Quality |
|-------|------|-------|------|---------|
| Llama-3.2-1B | 1B | ⚡⚡⚡ | 2GB | ⭐⭐⭐ |
| Llama-3.2-3B | 3B | ⚡⚡ | 6GB | ⭐⭐⭐⭐ |
| Llama-3.2-11B | 11B | ⚡ | 16GB | ⭐⭐⭐⭐⭐ |
| Llama-3-8B | 8B | ⚡ | 16GB | ⭐⭐⭐⭐⭐ |

## System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| CPU | 4 cores | 8+ cores |
| RAM | 8GB | 16GB+ |
| GPU VRAM | 0 (CPU only) | 8GB+ |
| Disk | 10GB | 20GB+ |
| Internet | 5Mbps | 20Mbps+ |

## Files to Know

| File | Purpose |
|------|---------|
| `team_page_scraper.py` | Main script |
| `team_requirements.txt` | Dependencies |
| `demo_prompt.txt` | Custom LLM prompt (optional) |
| `team_page/input1.csv` | Input URLs |
| `output/result_*.csv` | Final results |
| `raw_output/*.txt` | Raw cleaned text per site |
| `output/llm_raw/*.txt` | LLM response debug info |

## Key Script Functions

```python
# Read URLs
urls = read_urls_from_csv("team_page/input1.csv")

# Fetch and clean
site_doc = extract_site_documents(url)

# Initialize LLM
pipe = init_llm("meta-llama/Llama-3.2-3B")

# Extract people
people = ask_llm_for_extraction(pipe, cleaned_text, url)

# Write to CSV
append_csv_row(output_path, fieldnames, row)
```

## FAQ

**Q: Does it work with proxy?**  
A: Modify `fetch_url()` to add proxy support to requests

**Q: Can I use it with OpenAI API instead?**  
A: Yes, modify `init_llm()` and `ask_llm_for_extraction()` for OpenAI calls

**Q: How do I speed it up?**  
A: Use smaller model (1B) or batch process with GPU

**Q: Can I customize the prompt?**  
A: Yes! Create `demo_prompt.txt` in same directory, it will be used

**Q: Does it handle CAPTCHAs?**  
A: No, but you can pre-download HTML and modify script to read files

**Q: Can I resume from interruption?**  
A: Check existing CSV, manually remove last rows, re-run with different output file
