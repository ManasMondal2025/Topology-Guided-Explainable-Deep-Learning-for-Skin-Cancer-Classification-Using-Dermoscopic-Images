"""
config.py
---------
Central configuration for the Topology-Guided Explainable Skin Cancer
Classification project.

Every path below is relative to PROJECT_ROOT so the codebase can be moved
or deployed to a server without editing every script. Override any value
with an environment variable (see the `os.environ.get` fallbacks) or by
editing this file directly.
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Project root (this file's directory)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Dataset locations
# ---------------------------------------------------------------------------
# All raw datasets live under DATASETS_ROOT. See datasets/README.md for the
# exact download + extraction instructions for each dataset.
DATASETS_ROOT = Path(os.environ.get("TSC_DATASETS_ROOT", PROJECT_ROOT / "datasets"))

HAM10000_DIR = DATASETS_ROOT / "HAM10000"
HAM10000_METADATA_CSV = HAM10000_DIR / "HAM10000_metadata.csv"
HAM10000_IMAGE_DIRS = [
    HAM10000_DIR / "HAM10000_images_part_1",
    HAM10000_DIR / "HAM10000_images_part_2",
]

ISIC2018_DIR = DATASETS_ROOT / "ISIC2018"
ISIC2018_TRAIN_IMAGES = ISIC2018_DIR / "ISIC2018_Task1-2_Training_Input"
ISIC2018_TRAIN_MASKS = ISIC2018_DIR / "ISIC2018_Task1_Training_GroundTruth"
ISIC2018_VAL_IMAGES = ISIC2018_DIR / "ISIC2018_Task1-2_Validation_Input"
ISIC2018_VAL_MASKS = ISIC2018_DIR / "ISIC2018_Task1_Validation_GroundTruth"
ISIC2018_TEST_IMAGES = ISIC2018_DIR / "ISIC2018_Task1-2_Test_Input"

PH2_DIR = DATASETS_ROOT / "PH2"
PH2_DATASET_DIR = PH2_DIR / "PH2Dataset"  # contains one subfolder per lesion

# ---------------------------------------------------------------------------
# Checkpoints / outputs
# ---------------------------------------------------------------------------
CHECKPOINT_DIR = Path(os.environ.get("TSC_CHECKPOINT_DIR", PROJECT_ROOT / "checkpoints"))
OUTPUT_DIR = Path(os.environ.get("TSC_OUTPUT_DIR", PROJECT_ROOT / "outputs"))
LOG_DIR = OUTPUT_DIR / "logs"

UNET_CHECKPOINT = CHECKPOINT_DIR / "unet_segmentation_best.pth"
HYBRID_CHECKPOINT_BEST = CHECKPOINT_DIR / "hybrid_model_best.pth"
HYBRID_CHECKPOINT_LAST = CHECKPOINT_DIR / "hybrid_model_last.pth"

# U-Net encoder (segmentation/unet.py). "resnet34" is a good default
# accuracy/speed trade-off; bump to "resnet50" or "efficientnet-b3" if GPU
# budget allows (Sec. 10: RTX A4500 recommended). Full encoder zoo:
# https://smp.readthedocs.io/en/latest/encoders.html
UNET_ENCODER_NAME = os.environ.get("TSC_UNET_ENCODER", "resnet34")
UNET_ENCODER_WEIGHTS = "imagenet"

for d in [CHECKPOINT_DIR, OUTPUT_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# General
# ---------------------------------------------------------------------------
SEED = 42
IMAGE_SIZE = 224
NUM_CLASSES = 2  # benign vs malignant
CLASS_NAMES = ["benign", "malignant"]

# HAM10000 dx codes -> binary label mapping used throughout the project.
# mel (melanoma), bcc (basal cell carcinoma), akiec (actinic keratoses /
# intraepithelial carcinoma) are treated as malignant; the remaining
# diagnoses are treated as benign. This mapping is a common convention used
# for binary benign-vs-malignant reformulations of HAM10000.
MALIGNANT_DX = {"mel", "bcc", "akiec"}
BENIGN_DX = {"nv", "bkl", "df", "vasc"}

# ---------------------------------------------------------------------------
# Backbone
# ---------------------------------------------------------------------------
BACKBONE_NAME = os.environ.get("TSC_BACKBONE", "vit_b16")  # "vit_b16", "efficientnet_b3", or "mobilenet_v3"
VIT_MODEL_NAME = "vit_base_patch16_224"
EFFICIENTNET_MODEL_NAME = "tf_efficientnet_b3"
MOBILENET_MODEL_NAME = "mobilenetv3_large_100"  # matches the "MobileNet-V3" used in most skin-cancer literature
DEEP_FEATURE_DIM = 768  # output dim after backbone projection head

# ---------------------------------------------------------------------------
# Topology
# ---------------------------------------------------------------------------
TOPO_FEATURE_DIM = 34  # see topology/gudhi_features.py + topology/persistence.py
# 7 (betti + shape descriptors) + 25 (5x5 persistence image) + 2 (landscape mean/max) = 34
PERSISTENCE_IMAGE_RESOLUTION = (5, 5)
PERSISTENCE_LANDSCAPE_NUM_LANDSCAPES = 3
PERSISTENCE_LANDSCAPE_RESOLUTION = 50

# ---------------------------------------------------------------------------
# Fusion
# ---------------------------------------------------------------------------
FUSION_TYPE = os.environ.get("TSC_FUSION", "gated")  # "gated" or "attention"
FUSION_HIDDEN_DIM = 512

# ---------------------------------------------------------------------------
# Training hyperparameters
# ---------------------------------------------------------------------------
BATCH_SIZE = 32
NUM_WORKERS = 4
NUM_EPOCHS = 40
LEARNING_RATE = 3e-5
WEIGHT_DECAY = 1e-4
WARMUP_EPOCHS = 2
LABEL_SMOOTHING = 0.05
GRAD_CLIP_NORM = 1.0

TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

DEVICE = os.environ.get("TSC_DEVICE", "cuda")  # falls back to "cpu" automatically if unavailable
