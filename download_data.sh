#!/usr/bin/env bash
#
# download_data.sh
# ------------------
# Automates the pieces of dataset download that CAN be automated
# (ISIC 2018 via direct S3 links, HAM10000 via the Kaggle API).
# PH2 requires manual access-request approval from the ADDI project and is
# NOT automated here — see datasets/README.md Section 3.
#
# Usage:
#   bash download_data.sh isic2018          # ~13GB, no login required
#   bash download_data.sh ham10000          # requires ~/.kaggle/kaggle.json
#   bash download_data.sh all               # both of the above
#
# Run this from the project root (TopologySkinCancer/).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASETS_DIR="${TSC_DATASETS_ROOT:-$SCRIPT_DIR/datasets}"

download_isic2018() {
    echo "==> Downloading ISIC 2018 Task 1 (segmentation) to $DATASETS_DIR/ISIC2018 ..."
    mkdir -p "$DATASETS_DIR/ISIC2018"
    cd "$DATASETS_DIR/ISIC2018"

    base="https://isic-archive.s3.amazonaws.com/challenges/2018"
    files=(
        "ISIC2018_Task1-2_Training_Input.zip"
        "ISIC2018_Task1_Training_GroundTruth.zip"
        "ISIC2018_Task1-2_Validation_Input.zip"
        "ISIC2018_Task1_Validation_GroundTruth.zip"
    )
    for f in "${files[@]}"; do
        if [ ! -d "${f%.zip}" ]; then
            echo "  -> $f"
            wget -c "$base/$f"
            unzip -q -o "$f"
            rm "$f"
        else
            echo "  -> ${f%.zip} already extracted, skipping."
        fi
    done
    echo "==> ISIC 2018 Task 1 ready."
}

download_ham10000() {
    echo "==> Downloading HAM10000 (classification) to $DATASETS_DIR/HAM10000 ..."
    if ! command -v kaggle >/dev/null 2>&1; then
        echo "ERROR: the 'kaggle' CLI is not installed or not on PATH."
        echo "  pip install --break-system-packages kaggle"
        echo "Then place your API token at ~/.kaggle/kaggle.json (see datasets/README.md)."
        exit 1
    fi
    mkdir -p "$DATASETS_DIR/HAM10000"
    cd "$DATASETS_DIR/HAM10000"

    if [ ! -f "HAM10000_metadata.csv" ]; then
        kaggle datasets download -d kmader/skin-cancer-mnist-ham10000
        unzip -q -o skin-cancer-mnist-ham10000.zip -d .
        rm skin-cancer-mnist-ham10000.zip
    else
        echo "  -> HAM10000_metadata.csv already present, skipping."
    fi
    echo "==> HAM10000 ready."
}

case "${1:-}" in
    isic2018)
        download_isic2018
        ;;
    ham10000)
        download_ham10000
        ;;
    all)
        download_isic2018
        download_ham10000
        ;;
    *)
        echo "Usage: bash download_data.sh {isic2018|ham10000|all}"
        echo "(PH2 must be downloaded manually — see datasets/README.md Section 3)"
        exit 1
        ;;
esac
