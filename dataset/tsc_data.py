import torch
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from pytorch_lightning import LightningDataModule
from os.path import join

from .data_utils import forest_pretext_transform, center_xy_only, normalize_point_cloud

# Hardcoded mapping for site-specific ecoregions
# WRF, RMF -> 3E | NIF, OVF -> 5E
SITE_ECO_MAP = {"wrf_sp": "3E", "rmf_sp": "3E", "nif_sp": "5E", "ovf_sp": "5E"}

# Must match your Pre-training index exactly
ONTARIO_ECOREGIONS = ["2E", "2W", "3E", "3S", "3W", "4E", "4S", "4W", "5E", "5S", "6E"]
ECO_TO_IDX = {name: i for i, name in enumerate(ONTARIO_ECOREGIONS)}


class TSCDataset(Dataset):
    def __init__(self, files, dataset_name, rotate=False, transform=None):
        self.files = files
        self.transform = transform
        self.rotate = rotate

        # Determine ecoregion index from the site name
        eco_str = SITE_ECO_MAP.get(dataset_name, "3E")
        self.eco_idx = ECO_TO_IDX.get(eco_str, 0)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = np.load(self.files[idx], allow_pickle=True)
        coords = data["point_cloud"].astype(np.float32)  # (N, 3)
        label = data["label"].astype(np.float32)  # (num_species,)

        # 1. Height-preserving centering
        pc = center_xy_only(coords)

        # 2. Apply the same Stage A transformations
        if self.transform:
            # Reusing your pointCloudTransform function
            pc, _, _ = forest_pretext_transform(
                pc, pc_feat=None, target=None, rot=self.rotate
            )
        pos = pc.copy()
        # 3. Features: Normalized coordinates
        x = normalize_point_cloud(pc)

        return {
            "point_cloud": torch.from_numpy(pos).float(),  # 'pos' for grouping
            "pc_feat": torch.from_numpy(x).float(),  # 'x' as input feature
            "label": torch.from_numpy(label).float(),  # Composition vector
            "ecoregion": torch.tensor(self.eco_idx, dtype=torch.long),
        }


class TSCDataModule(LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.batch_size = config["batch_size"]
        self.num_workers = config.get("num_workers", 6)

        # Site name (e.g. 'wrf_sp') used for ecoregion mapping
        self.dataset_name = config["dataset"]

        self.data_dirs = {
            "train": join(config["data_dir"], "tile_128", "train", self.dataset_name),
            "val": join(config["data_dir"], "tile_128", "val", self.dataset_name),
            "test": join(
                config.get("test_data_dir", config["data_dir"]),
                "tile_128",
                "test",
                self.dataset_name,
            ),
        }

    def _get_files(self, split):
        d = self.data_dirs[split]
        if not os.path.exists(d):
            return []
        return [join(d, f) for f in os.listdir(d) if f.endswith(".npz")]

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            train_files = self._get_files("train")
            val_files = self._get_files("val")

            # Clean Dataset
            train_ds = TSCDataset(train_files, self.dataset_name, rotate=False)

            # Augmented Dataset (Site-specific tuning often needs more augs)
            if self.config.get("point_cloud_transform"):
                aug_ds = TSCDataset(
                    train_files,
                    self.dataset_name,
                    rotate=self.config["rotate"],
                    transform=self.config["point_cloud_transform"],
                )
                self.train_dataset = ConcatDataset([train_ds, aug_ds])
            else:
                self.train_dataset = train_ds

            self.val_dataset = TSCDataset(val_files, self.dataset_name)

        if stage == "test" or stage is None:
            self.test_dataset = TSCDataset(self._get_files("test"), self.dataset_name)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
