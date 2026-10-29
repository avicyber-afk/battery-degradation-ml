#!/bin/bash
set -e
declare -A FILES=(
  ["MATR_batch_20170512.mat"]="https://data.matr.io/1/api/v1/file/5c86c0b5fa2ede00015ddf66/download"
  ["MATR_batch_20170630.mat"]="https://data.matr.io/1/api/v1/file/5c86bf13fa2ede00015ddd82/download"
  ["MATR_batch_20180412.mat"]="https://data.matr.io/1/api/v1/file/5c86bd64fa2ede00015ddbb2/download"
  ["MATR_batch_20190124.mat"]="https://data.matr.io/1/api/v1/file/5dcef152110002c7215b2c90/download"
)
for name in "${!FILES[@]}"; do
  echo "=== Downloading $name ==="
  curl -L -o "$name" "${FILES[$name]}" --progress-bar
  echo "=== Done: $name ($(du -h "$name" | cut -f1)) ==="
done
echo "ALL_DOWNLOADS_COMPLETE"
