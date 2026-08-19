# Dataset Download & Placement Guide

This project expects all three datasets from Sec. 7 of the proposal (HAM10000,
ISIC 2018, PH2) under `datasets/`, in the exact layout below. `config.py`
already points at these paths, so if you follow this layout nothing else
needs to be changed.

```
TopologySkinCancer/
└── datasets/
    ├── HAM10000/
    │   ├── HAM10000_metadata.csv
    │   ├── HAM10000_images_part_1/         (5000 .jpg images)
    │   ├── HAM10000_images_part_2/         (5015 .jpg images)
    │   └── predicted_masks/                (auto-generated later, see Step 4)
    │
    ├── ISIC2018/
    │   ├── ISIC2018_Task1-2_Training_Input/       (2594 .jpg images)
    │   ├── ISIC2018_Task1_Training_GroundTruth/   (2594 _segmentation.png masks)
    │   ├── ISIC2018_Task1-2_Validation_Input/     (100 .jpg images)
    │   ├── ISIC2018_Task1_Validation_GroundTruth/ (100 masks)
    │   ├── ISIC2018_Task1-2_Test_Input/           (1000 .jpg images, optional)
    │   └── ISIC2018_Task1_Test_GroundTruth/       (optional)
    │
    └── PH2/
        └── PH2Dataset/
            ├── PH2_dataset.txt
            └── PH2 Dataset images/
                ├── IMD002/
                │   ├── IMD002_Dermoscopic_Image/IMD002.bmp
                │   └── IMD002_lesion/IMD002_lesion.bmp
                ├── IMD003/
                └── ... (200 cases total)
```

If your server stores large datasets on a different disk/mount (very common
on shared GPU servers, e.g. `/data/` or `/mnt/data/`), you do **not** need to
move anything into the repo — just point `config.py` at it instead:

```bash
export TSC_DATASETS_ROOT=/data/TopologySkinCancer_datasets
```

(`config.py` reads `TSC_DATASETS_ROOT` if set, otherwise defaults to
`<project_root>/datasets`.)

---

## 1. HAM10000 (classification — Sec. 7, Dataset 1)

**Option A — Kaggle (recommended, fastest on a server):**

```bash
pip install --break-system-packages kaggle
# Get an API token from https://www.kaggle.com/settings -> "Create New Token"
# this downloads kaggle.json; place it at ~/.kaggle/kaggle.json (chmod 600)
mkdir -p ~/.kaggle && mv kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json

cd TopologySkinCancer/datasets/HAM10000
kaggle datasets download -d kmader/skin-cancer-mnist-ham10000
unzip skin-cancer-mnist-ham10000.zip -d .
# The Kaggle mirror nests images under HAM10000_images_part_1/2 folders and
# ships HAM10000_metadata.csv at the top level already matching the layout above.
rm skin-cancer-mnist-ham10000.zip
```

**Option B — official Harvard Dataverse record (source of truth):**
`https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T`
Click "Access Dataset" -> "Original Format ZIP", accept the terms of use (this
step requires a browser, Dataverse does not expose a stable anonymous direct
link), download to your machine, then `scp`/`rsync` the zip to the server and
`unzip` it into `datasets/HAM10000/` following the layout above.

This dataset is for **non-commercial research use only** — read the license
on the Dataverse page before using it beyond this MSc project.

---

## 2. ISIC 2018 Task 1 (segmentation — Sec. 7, Dataset 2)

The official ISIC challenge archive serves these directly from S3, so `wget`
works with no login:

```bash
cd TopologySkinCancer/datasets/ISIC2018

wget https://isic-archive.s3.amazonaws.com/challenges/2018/ISIC2018_Task1-2_Training_Input.zip
wget https://isic-archive.s3.amazonaws.com/challenges/2018/ISIC2018_Task1_Training_GroundTruth.zip
wget https://isic-archive.s3.amazonaws.com/challenges/2018/ISIC2018_Task1-2_Validation_Input.zip
wget https://isic-archive.s3.amazonaws.com/challenges/2018/ISIC2018_Task1_Validation_GroundTruth.zip
# optional test split (no public ground truth needed for this project):
wget https://isic-archive.s3.amazonaws.com/challenges/2018/ISIC2018_Task1-2_Test_Input.zip

unzip ISIC2018_Task1-2_Training_Input.zip
unzip ISIC2018_Task1_Training_GroundTruth.zip
unzip ISIC2018_Task1-2_Validation_Input.zip
unzip ISIC2018_Task1_Validation_GroundTruth.zip
rm *.zip
```

Training Input is ~10.4GB, so make sure you have disk space (and use `tmux`/
`screen`/`nohup` since it takes a while on most connections). This data is
released under **CC-0** by ISIC, so no account/license gate.

Licensed reference: `https://challenge.isic-archive.com/data/`

---

## 3. PH2 (external validation — Sec. 7, Dataset 3)

PH2 is distributed by the ADDI project (Universidade do Porto) and its
official page requires filling in a short access-request form rather than
exposing a stable public download URL:

`https://www.fc.up.pt/addi/ph2%20database.html`

1. Visit the page above, follow the request-access instructions, and
   download `PH2Dataset.zip` / `.rar`.
2. Transfer it to the server and extract it so the final path is
   `datasets/PH2/PH2Dataset/PH2 Dataset images/IMDxxx/...` (see the tree
   above). If the archive extracts to a differently-named top folder,
   rename/move it — `data/datasets.py::PH2Dataset` searches recursively for a
   folder literally named `PH2 Dataset images`, so nesting depth doesn't
   matter as long as that folder name is preserved.

```bash
mkdir -p TopologySkinCancer/datasets/PH2
cd TopologySkinCancer/datasets/PH2
# after transferring PH2Dataset.zip here:
unzip PH2Dataset.zip -d PH2Dataset
rm PH2Dataset.zip
```

Several unofficial Kaggle mirrors also exist (e.g. `kliuiev/ph2databaseaddi`,
`ramzesii/ph2dataset`) if you prefer `kaggle datasets download`, but treat the
ADDI page as the source of truth for licensing/attribution requirements —
this project uses PH2 only for read-only external validation, never training.

---

## 4. Generate the masks HAM10000 is missing

HAM10000 ships **no** ground-truth segmentation masks, but the topology
pipeline needs one lesion mask per image. Train U-Net on ISIC 2018 first,
then run it once over HAM10000 (and PH2, as a cross-check against its own
ground-truth masks) to cache predicted masks to disk:

```bash
cd TopologySkinCancer
python -m segmentation.train_unet --epochs 60          # writes checkpoints/unet_segmentation_best.pth
python -m segmentation.predict_masks --dataset ham10000  # writes datasets/HAM10000/predicted_masks/
python -m segmentation.predict_masks --dataset ph2       # writes datasets/PH2/predicted_masks/ (optional)
```

After this step, `datasets/HAM10000/predicted_masks/` contains one
`<image_id>_segmentation.png` per HAM10000 image, and you're ready to run
`train.py` (see the top-level `README.md`).
