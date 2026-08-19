"""
data/topo_wrapper.py
-----------------------
Wraps any of the classification datasets in data/datasets.py (HAM10000,
PH2) so that each sample also carries its fixed-length topological
feature vector (Sec. 6, Phase 4 + Phase 6), computed on-the-fly from the
sample's lesion mask via topology/extract.py.

Kept as a separate wrapper (rather than baked into datasets.py) so the
same underlying datasets can also be used stand-alone for segmentation-only
experiments without paying the topology-extraction cost.
"""

import logging

import numpy as np
import torch
from torch.utils.data import Dataset

from topology.extract import extract_topological_features, get_topology_feature_dim


class TopologyAugmentedDataset(Dataset):
    """
    base_dataset: any Dataset whose __getitem__ returns a dict containing
                  at least {"image": Tensor, "mask": Tensor[1,H,W], "label": int, "image_id": str}
    """

    def __init__(self, base_dataset: Dataset):
        self.base_dataset = base_dataset
        self.topo_dim = get_topology_feature_dim()

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        sample = self.base_dataset[idx]
        mask_np = sample["mask"].squeeze(0).cpu().numpy().astype(np.uint8) * 255

        try:
            topo_features = extract_topological_features(mask_np)
        except Exception as e:  # noqa: BLE001 - guard against pathological empty masks
            logging.warning(f"Topology extraction failed for {sample.get('image_id')}: {e}. "
                             "Falling back to a zero vector.")
            topo_features = np.zeros(self.topo_dim, dtype=np.float32)

        sample["topo_features"] = torch.from_numpy(topo_features).float()
        return sample


if __name__ == "__main__":
    # smoke test with a synthetic in-memory dataset
    class _DummyDataset(Dataset):
        def __len__(self):
            return 3

        def __getitem__(self, idx):
            import cv2
            mask = np.zeros((224, 224), dtype=np.uint8)
            cv2.circle(mask, (112, 112), 60 + idx * 5, 1, -1)
            return {
                "image": torch.randn(3, 224, 224),
                "mask": torch.from_numpy(mask).unsqueeze(0).float(),
                "label": idx % 2,
                "image_id": f"dummy_{idx}",
            }

    wrapped = TopologyAugmentedDataset(_DummyDataset())
    sample = wrapped[0]
    print("topo_features shape:", sample["topo_features"].shape)
    assert sample["topo_features"].shape[0] == wrapped.topo_dim
    print("OK")
