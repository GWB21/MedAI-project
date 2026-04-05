#!/bin/bash
# PMC-VQA dataset download script
# Downloads test_clean.csv and associated images from HuggingFace
set -e

DATA_DIR="./data/pmc_vqa"
mkdir -p "$DATA_DIR"

echo "=== PMC-VQA Dataset Download ==="
echo "Target directory: $DATA_DIR"

# Check for huggingface-cli
if ! command -v huggingface-cli &> /dev/null; then
    echo "Installing huggingface_hub..."
    pip install -q huggingface_hub
fi

# Download test_clean.csv
echo ""
echo "[1/3] Downloading test_clean.csv..."
huggingface-cli download xmcmic/PMC-VQA test_clean.csv \
    --repo-type dataset \
    --local-dir "$DATA_DIR"

# Download images (version 1 -- compound images matching test_clean)
echo ""
echo "[2/3] Downloading images..."
echo "  This may take a while (~15GB)..."
huggingface-cli download xmcmic/PMC-VQA images.zip \
    --repo-type dataset \
    --local-dir "$DATA_DIR"

# Extract images
if [ -f "$DATA_DIR/images.zip" ]; then
    echo "  Extracting images.zip..."
    unzip -q -o "$DATA_DIR/images.zip" -d "$DATA_DIR"
    echo "  Done."
fi

# Verify
echo ""
echo "[3/3] Verifying..."
python3 -c "
import pandas as pd
import os

csv_path = '$DATA_DIR/test_clean.csv'
df = pd.read_csv(csv_path)
print(f'  CSV rows: {len(df)}')
print(f'  Columns: {list(df.columns)}')

# Check image availability
found = 0
missing_examples = []
for _, row in df.iterrows():
    fig = row['Figure_path']
    candidates = [
        os.path.join('$DATA_DIR', fig),
        os.path.join('$DATA_DIR', 'images', fig),
        os.path.join('$DATA_DIR', 'images', os.path.basename(fig)),
    ]
    if any(os.path.exists(c) for c in candidates):
        found += 1
    elif len(missing_examples) < 3:
        missing_examples.append(fig)

print(f'  Images found: {found}/{len(df)}')
if missing_examples:
    print(f'  Missing examples: {missing_examples}')

if found == len(df):
    print('  ALL IMAGES FOUND')
elif found == 0:
    print('  WARNING: No images found. Check image directory structure.')
"

echo ""
echo "=== Download Complete ==="
echo "Data directory: $DATA_DIR"
