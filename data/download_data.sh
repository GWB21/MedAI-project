#!/bin/bash
# PMC-VQA test_clean set download
# Only downloads test_clean.csv + required images (not the full dataset)
set -e

DATA_DIR="./data/pmc_vqa"
mkdir -p "$DATA_DIR"

echo "=== PMC-VQA test_clean Download ==="
echo "Target: $DATA_DIR"

# Check for huggingface-cli
if ! command -v huggingface-cli &> /dev/null; then
    echo "Installing huggingface_hub..."
    pip install -q huggingface_hub
fi

# 1. Download test_clean.csv only
echo ""
echo "[1/3] Downloading test_clean.csv..."
huggingface-cli download xmcmic/PMC-VQA test_clean.csv \
    --repo-type dataset \
    --local-dir "$DATA_DIR"

# 2. Download images (needed for test_clean)
# test_clean uses images from images.zip (version 1, compound images)
echo ""
echo "[2/3] Downloading images for test_clean..."
echo "  Note: Only images.zip is needed (not images2.zip)."
huggingface-cli download xmcmic/PMC-VQA images.zip \
    --repo-type dataset \
    --local-dir "$DATA_DIR"

# Extract
if [ -f "$DATA_DIR/images.zip" ]; then
    echo "  Extracting images.zip..."
    unzip -q -o "$DATA_DIR/images.zip" -d "$DATA_DIR"
    echo "  Cleaning up zip..."
    rm -f "$DATA_DIR/images.zip"
fi

# 3. Verify
echo ""
echo "[3/3] Verifying test_clean data..."
python3 -c "
import pandas as pd
import os

csv_path = '$DATA_DIR/test_clean.csv'
df = pd.read_csv(csv_path)
print(f'  Samples: {len(df)}')
print(f'  Columns: {list(df.columns)}')

found = 0
for _, row in df.iterrows():
    fig = row['Figure_path']
    candidates = [
        os.path.join('$DATA_DIR', fig),
        os.path.join('$DATA_DIR', 'images', fig),
        os.path.join('$DATA_DIR', 'images', os.path.basename(fig)),
    ]
    if any(os.path.exists(c) for c in candidates):
        found += 1

print(f'  Images: {found}/{len(df)}')
if found == len(df):
    print('  Status: ALL OK')
elif found == 0:
    print('  WARNING: No images found. Check directory structure.')
    print('  Expected: $DATA_DIR/images/<Figure_path>')
else:
    print(f'  WARNING: {len(df) - found} images missing')
"

echo ""
echo "=== Download Complete ==="
