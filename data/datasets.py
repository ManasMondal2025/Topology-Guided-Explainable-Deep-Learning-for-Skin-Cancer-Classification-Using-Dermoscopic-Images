"""
data/datasets.py
-----------------
PyTorch Dataset classes for the three datasets used in the proposal
(Sec. 7): HAM10000 (classification), ISIC 2018 (segmentation + topology
extraction) and PH2 (external validation).

Each classification-style dataset returns a dict:
    {
        "image": Tensor[3, H, W],
        "mask":  Tensor[1, H, W]  (predicted or ground-truth lesion mask),
        "label": int (0=benign, 1=malignant),
        "image_id": str,
    }

so that the same training/eval loop can consume any of them.
"""

import os
import glob
import logging
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

import config


def _normalize_mask(mask) -> torch.Tensor:
    """
    Converts a mask (numpy array OR already-a-tensor, coming out of an
    Albumentations transform) into a canonical Tensor[1, H, W] of {0., 1.}.
    Handles both the "no transform" (numpy, values in {0, 255}) and
    "ToTensorV2 applied" (tensor, still values in {0, 255} or {0, 1}) cases.
    """
    if not torch.is_tensor(mask):
        mask = torch.as_tensor(np.asarray(mask))
    mask = mask.float()
    if mask.max() > 1.0:
        mask = mask / 255.0
    mask = (mask > 0.5).float()
    if mask.dim() == 2:
        mask = mask.unsqueeze(0)
    return mask


# ---------------------------------------------------------------------------
# HAM10000 - classification
# ---------------------------------------------------------------------------
class HAM10000Dataset(Dataset):
    """
    Expects:
        datasets/HAM10000/HAM10000_metadata.csv
        datasets/HAM10000/HAM10000_images_part_1/*.jpg
        datasets/HAM10000/HAM10000_images_part_2/*.jpg

    dx column values are mapped to benign(0)/malignant(1) using
    config.MALIGNANT_DX / config.BENIGN_DX.
    """

    def __init__(self, metadata_csv=None, image_dirs=None, transform=None,
                 mask_dir=None, mask_transform=None, indices=None):
        self.metadata_csv = Path(metadata_csv or config.HAM10000_METADATA_CSV)
        self.image_dirs = [Path(d) for d in (image_dirs or config.HAM10000_IMAGE_DIRS)]
        self.transform = transform
        self.mask_dir = Path(mask_dir) if mask_dir is not None else None
        self.mask_transform = mask_transform

        if not self.metadata_csv.exists():
            raise FileNotFoundError(
                f"HAM10000 metadata CSV not found at {self.metadata_csv}. "
                "See datasets/README.md for download instructions."
            )

        df = pd.read_csv(self.metadata_csv)
        df = df[df["dx"].isin(config.MALIGNANT_DX | config.BENIGN_DX)].reset_index(drop=True)
        df["label"] = df["dx"].apply(lambda dx: 1 if dx in config.MALIGNANT_DX else 0)

        # HAM10000 contains multiple images of the SAME physical lesion
        # (follow-up photos). Splitting at the image level risks the same
        # lesion appearing in both train and test, which leaks information
        # and inflates test metrics. `lesion_id` is used for GroupShuffleSplit
        # in train.py/evaluate.py; if a metadata CSV lacks the column (e.g. a
        # minimal/synthetic fixture), fall back to one group per image.
        if "lesion_id" not in df.columns:
            df["lesion_id"] = df["image_id"]

        # Resolve each image_id to its actual file path (images are split
        # across two folders in the official release).
        df["image_path"] = df["image_id"].apply(self._resolve_image_path)
        df = df[df["image_path"].notna()].reset_index(drop=True)

        if indices is not None:
            df = df.iloc[indices].reset_index(drop=True)

        self.df = df
        logging.info(f"HAM10000Dataset: {len(self.df)} usable samples "
                      f"({(self.df['label']==1).sum()} malignant / {(self.df['label']==0).sum()} benign)")

    def _resolve_image_path(self, image_id: str):
        for d in self.image_dirs:
            p = d / f"{image_id}.jpg"
            if p.exists():
                return str(p)
        return None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = cv2.cvtColor(cv2.imread(row["image_path"]), cv2.COLOR_BGR2RGB)

        mask = None
        if self.mask_dir is not None:
            mask_path = self.mask_dir / f"{row['image_id']}_segmentation.png"
            if mask_path.exists():
                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        if mask is None:
            # No ground-truth mask available for HAM10000; a placeholder of
            # zeros is returned and is expected to be replaced by the
            # U-Net segmentation model's prediction at train/inference time
            # (see train.py: `predict_mask_if_missing`).
            mask = np.zeros(image.shape[:2], dtype=np.uint8)

        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image, mask = augmented["image"], augmented["mask"]
        mask = _normalize_mask(mask)

        return {
            "image": image,
            "mask": mask,
            "label": int(row["label"]),
            "image_id": row["image_id"],
        }


# ---------------------------------------------------------------------------
# ISIC 2018 - segmentation (Task 1) + optional classification labels (Task 3)
# ---------------------------------------------------------------------------
class ISIC2018SegmentationDataset(Dataset):
    """
    Expects the official ISIC 2018 Task 1 structure:
        datasets/ISIC2018/ISIC2018_Task1-2_Training_Input/ISIC_xxxxxxx.jpg
        datasets/ISIC2018/ISIC2018_Task1_Training_GroundTruth/ISIC_xxxxxxx_segmentation.png
    """

    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.transform = transform

        if not self.image_dir.exists():
            raise FileNotFoundError(
                f"ISIC2018 image directory not found at {self.image_dir}. "
                "See datasets/README.md for download instructions."
            )

        self.image_paths = sorted(glob.glob(str(self.image_dir / "*.jpg")))
        logging.info(f"ISIC2018SegmentationDataset: {len(self.image_paths)} images found")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image_id = Path(image_path).stem
        mask_path = self.mask_dir / f"{image_id}_segmentation.png"

        image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Missing ground-truth mask: {mask_path}")

        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image, mask = augmented["image"], augmented["mask"]

        mask = _normalize_mask(mask)

        return {"image": image, "mask": mask, "image_id": image_id}


# ---------------------------------------------------------------------------
# PH2 - external validation only
# ---------------------------------------------------------------------------
class PH2Dataset(Dataset):
    """
    Expects the official PH2 folder layout:
        datasets/PH2/PH2Dataset/PH2 Dataset images/IMDxxx/IMDxxx_Dermoscopic_Image/IMDxxx.bmp
        datasets/PH2/PH2Dataset/PH2 Dataset images/IMDxxx/IMDxxx_lesion/IMDxxx_lesion.bmp
        datasets/PH2/PH2Dataset/PH2_dataset.txt   (metadata with diagnosis)

    PH2_dataset.txt is a '||'-delimited table (verified against the real
    file, not assumed from general PH2 documentation). A data row looks
    like:
        || IMD003 ||                        ||                  0 || ...
    Splitting on '||' gives (after stripping whitespace):
        [0]="", [1]=Name, [2]=Histological Diagnosis, [3]=Clinical Diagnosis,
        [4]=<ABCD features, further '|'-delimited>, [5]=Colors, [6]=""
    "Clinical Diagnosis" is a NUMERIC code, not text:
        0 = Common Nevus (benign), 1 = Atypical Nevus (benign), 2 = Melanoma (malignant)
    An earlier version of this parser searched for the literal words
    "Melanoma"/"Common Nevus"/"Atypical Nevus" in each line -- those words
    never appear in the actual per-case data rows (diagnosis is numeric),
    so that version silently failed to label any benign case, making
    every PH2 evaluation's precision/specificity/F1/confusion-matrix
    meaningless (only malignant cases got a label at all). Fixed here to
    parse the real numeric-code format directly.
    """

    def __init__(self, root_dir=None, transform=None):
        self.root_dir = Path(root_dir or config.PH2_DATASET_DIR)
        self.transform = transform

        images_root = self._find_images_root()
        if images_root is None:
            raise FileNotFoundError(
                f"PH2 dataset not found under {self.root_dir}. "
                "See datasets/README.md for download instructions."
            )
        self.images_root = images_root
        self.case_ids = sorted(
            [p.name for p in self.images_root.iterdir() if p.is_dir() and p.name.startswith("IMD")]
        )
        self.labels = self._load_labels()
        logging.info(f"PH2Dataset: {len(self.case_ids)} cases found")

    def _find_images_root(self):
        candidates = list(self.root_dir.rglob("PH2 Dataset images"))
        return candidates[0] if candidates else None

    def _load_labels(self):
        """Parses PH2_dataset.txt's numeric 'Clinical Diagnosis' column
        (field index 3 after splitting each line on '||') -- see class
        docstring for the verified real file format."""
        labels = {}
        txt_candidates = list(self.root_dir.rglob("PH2_dataset.txt"))
        if not txt_candidates:
            logging.warning("PH2_dataset.txt not found; labels default to -1 (unknown) "
                             "and this split can only be used for segmentation/topology checks.")
            return labels

        with open(txt_candidates[0], "r", errors="ignore") as f:
            for line in f:
                if "||" not in line:
                    continue
                fields = [p.strip() for p in line.split("||")]
                if len(fields) < 4:
                    continue
                case_id = fields[1]
                if not case_id.startswith("IMD"):
                    continue  # skips the header row and any non-data lines
                diagnosis_str = fields[3]
                if diagnosis_str not in ("0", "1", "2"):
                    continue  # defensive: skip malformed/unexpected rows
                diagnosis_code = int(diagnosis_str)
                labels[case_id] = 1 if diagnosis_code == 2 else 0

        # Sanity-check the parse result rather than failing silently: PH2 is
        # documented to be ~80 Common Nevus + ~80 Atypical Nevus + ~40
        # Melanoma (i.e. both classes should be well represented). If a
        # future PH2 distribution changes format again and parsing breaks,
        # this makes it loud instead of silently producing a
        # single-class-only label set (which is what happened before this
        # fix -- see class docstring).
        n_benign = sum(1 for v in labels.values() if v == 0)
        n_malignant = sum(1 for v in labels.values() if v == 1)
        logging.info(f"PH2 labels parsed: {len(labels)} labeled "
                      f"({n_benign} benign, {n_malignant} malignant)")
        if labels and (n_benign == 0 or n_malignant == 0):
            logging.error(
                f"PH2 label parsing looks BROKEN: {n_benign} benign / {n_malignant} malignant "
                "labels found (expected both classes well represented). Metrics computed on "
                "PH2 (precision, specificity, F1, confusion matrix) will be meaningless. "
                "Inspect PH2_dataset.txt directly before trusting any PH2 evaluation results."
            )
        return labels

    def __len__(self):
        return len(self.case_ids)

    def __getitem__(self, idx):
        case_id = self.case_ids[idx]
        case_dir = self.images_root / case_id
        image_path = case_dir / f"{case_id}_Dermoscopic_Image" / f"{case_id}.bmp"
        mask_path = case_dir / f"{case_id}_lesion" / f"{case_id}_lesion.bmp"

        image = cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) if mask_path.exists() else \
            np.zeros(image.shape[:2], dtype=np.uint8)

        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image, mask = augmented["image"], augmented["mask"]
        mask = _normalize_mask(mask)

        label = self.labels.get(case_id, -1)
        return {"image": image, "mask": mask, "label": label, "image_id": case_id}
