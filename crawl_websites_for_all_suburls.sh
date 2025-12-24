#!/bin/bash
set -euo pipefail

######################################
# CONFIG
######################################
INPUT_FILE="websites.csv"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_FILE="suburls_${TIMESTAMP}.csv"
TMP_FILE="/tmp/crawl_urls_${TIMESTAMP}.tmp"
BATCH_SIZE=100

######################################
# INIT
######################################
echo "url" > "$OUTPUT_FILE"
: > "$TMP_FILE"

total_urls=0
site_count=0
batch_count=0

######################################
# PROCESS
######################################
tail -n +2 "$INPUT_FILE" | while IFS=',' read -r site; do
  site=$(echo "$site" | xargs)
  [ -z "$site" ] && continue

  ((site_count++))
  echo "[INFO] [$site_count] Crawling: $site"

  urls=$(curl -sL "$site" \
    | grep -Eo 'href="https?://[^"]+"' \
    | sed 's/href="//;s/"//' \
    | grep -E "^$site" || true)

  if [ -n "$urls" ]; then
    while read -r url; do
      echo "$url" >> "$TMP_FILE"
      ((batch_count++))
      ((total_urls++))

      if (( batch_count >= BATCH_SIZE )); then
        cat "$TMP_FILE" >> "$OUTPUT_FILE"
        : > "$TMP_FILE"
        batch_count=0
        echo "[INFO] URLs written so far: $total_urls"
      fi
    done <<< "$urls"
  else
    echo "[WARN] No URLs found for $site"
  fi

done

######################################
# FLUSH REMAINING
######################################
if [ -s "$TMP_FILE" ]; then
  cat "$TMP_FILE" >> "$OUTPUT_FILE"
  echo "[INFO] Final flush completed"
fi

rm -f "$TMP_FILE"

######################################
# FINAL STATS
######################################
echo "===================================="
echo "Done."
echo "Output file : $OUTPUT_FILE"
echo "Sites crawled: $site_count"
echo "Total subURLs   : $total_urls"
echo "===================================="
