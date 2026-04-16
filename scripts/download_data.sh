#!/bin/bash
# Download datasets for chemprop-redux
set -e

DATA_DIR="$(cd "$(dirname "$0")/../data" && pwd)"
mkdir -p "$DATA_DIR"

# COCONUT natural products (eval dataset)
COCONUT_CSV="$DATA_DIR/coconut_csv_lite-04-2026.csv"
if [ ! -f "$COCONUT_CSV" ]; then
    echo "Downloading COCONUT CSV lite (~191 MB)..."
    curl -L "https://coconut.s3.uni-jena.de/prod/downloads/2026-04/coconut_csv_lite-04-2026.zip" \
        -o "$DATA_DIR/coconut_csv_lite.zip"
    unzip -o "$DATA_DIR/coconut_csv_lite.zip" -d "$DATA_DIR"
    rm "$DATA_DIR/coconut_csv_lite.zip"
    echo "COCONUT data saved to $COCONUT_CSV"
else
    echo "COCONUT data already exists, skipping."
fi

echo "Done."
