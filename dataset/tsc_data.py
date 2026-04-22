import torch
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from pytorch_lightning import LightningDataModule
from os.path import join

from .data_utils import (
    forest_pretext_transform,
    center_point_cloud,
    normalize_point_cloud,
)

# Hardcoded mapping for site-specific ecoregions
# WRF, RMF -> 3E | NIF, OVF -> 5E
SITE_ECO_MAP = {
    "wrf_sp": "3E",
    "rmf_sp": "3E",
    "nif_sp": "5E",
    "ovf_sp": "5E",
    "ovf_sub": "5E",
}

# Must match your Pre-training index exactly
ONTARIO_ECOREGIONS = ["2E", "2W", "3E", "3S", "3W", "4E", "4S", "4W", "5E", "5S", "6E"]
ECO_TO_IDX = {name: i for i, name in enumerate(ONTARIO_ECOREGIONS)}


class TSCDataset(Dataset):
    def __init__(self, files, dataset_name, embed_dir=None, transform=None):
        self.files = files
        self.transform = transform
        self.embed_dir = embed_dir

        # Determine ecoregion index
        eco_str = SITE_ECO_MAP.get(dataset_name, "3E")
        self.eco_idx = ECO_TO_IDX.get(eco_str, 0)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # Load point cloud data
        data = np.load(self.files[idx], allow_pickle=True)
        coords = data["point_cloud"]
        label = data["label"]

        # --- NEW: Load Patch Embedding ---
        # Assuming the .npz filename is 'POLYID.npz' or it has a 'polyid' key
        # If filename is e.g., "plot_1234.npz", we extract "1234"
        polyid = os.path.basename(self.files[idx]).replace(".npz", "")

        patch_embed = torch.zeros((3, 3, 128))  # Default fallback
        if self.embed_dir:
            embed_path = join(self.embed_dir, f"{polyid}.npy")
            if os.path.exists(embed_path):
                # Load (3, 3, 128) array
                patch_arr = np.load(embed_path)
                patch_embed = torch.from_numpy(patch_arr).float()

        # 1. Height-preserving centering
        pc = center_point_cloud(coords)

        # 2. Features: Normalized coordinates
        feats = normalize_point_cloud(pc)

        # 3. Apply transformations
        if self.transform:
            pc, feats, label = forest_pretext_transform(
                pc, pc_feat=feats, target=label, rot=False
            )

        return {
            "point_cloud": torch.from_numpy(pc).float(),
            "pc_feat": torch.from_numpy(feats).float(),
            "label": torch.from_numpy(label).float(),
            "ecoregion": torch.tensor(self.eco_idx, dtype=torch.long),
            "patch_embed": patch_embed,  # <--- Added to return dict
        }


class TSCDataModule(LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.batch_size = config["batch_size"]
        self.num_workers = 2
        self.dataset_name = config["dataset"]

        # Path where your sampled .npy files are stored
        self.embed_dir = config.get(
            "img_emb_dir",
            f"./data/{self.dataset_name.split('_')[0]}_img/tessera_tiles/{self.dataset_name.split('_')[0]}_embeddings",
        )

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

            # Pass embed_dir to datasets
            train_ds = TSCDataset(
                train_files, self.dataset_name, embed_dir=self.embed_dir
            )
            aug_ds = TSCDataset(
                train_files, self.dataset_name, embed_dir=self.embed_dir, transform=True
            )

            self.train_dataset = ConcatDataset([train_ds, aug_ds])
            self.val_dataset = TSCDataset(
                val_files, self.dataset_name, embed_dir=self.embed_dir
            )

        if stage == "test" or stage is None:
            self.test_dataset = TSCDataset(
                self._get_files("test"), self.dataset_name, embed_dir=self.embed_dir
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=False,
        )
