import pandas as pd
import numpy as np
import os

import torch
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule

from .data_utils import forest_pretext_transform

ONTARIO_ECOREGIONS = [
    "2E",
    "2W",
    "3E",
    "3S",
    "3W",
    "4E",
    "4S",
    "4W",
    "5E",
    "5S",
    "6E",
]

ONTARIO_SPECIES = [2, 5, 6, 9, 10, 12, 13, 17, 18, 22, 25, 26, 28, 29, 31, 32]

class OntarioPretextDataset(Dataset):

    def __init__(
        self,
        manifest_df,
        data_root,
        eco_column="ecoregion_idx",
        transform=None,
        rotate=False,
    ):
        """
        Args:
            manifest_df: Dataframe containing [relative_path, label, h95, ecoregion]
            data_root: Base folder where .npy files are stored
            transform: Point cloud augmentation function
        """
        self.df = manifest_df
        self.data_root = data_root
        self.transform = transform
        self.eco_column = eco_column
        self.rot = rotate
        self.num_points = 7168

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # 1. Load raw extracted points (centered XY, absolute Z meters)
        # Shape: (N, 3) -> [X_rel, Y_rel, Z_hag]
        pc_raw = np.load(os.path.join(self.data_root, row['relative_path'])).astype(np.float32)

        n_points = pc_raw.shape[0]

        if n_points >= self.num_points:
            # Downsample: Pick a random subset
            idx_pts = np.random.choice(n_points, self.num_points, replace=False)
            pc_raw = pc_raw[idx_pts]

        # 2. Augmentation (Rotation/Scaling/Jitter)
        # Apply these to the 'raw' meter-scale points first
        if self.transform:
            pc_raw, _, _ = forest_pretext_transform(
                pc_raw, pc_feat=None, target=None, rot=self.rot
            )

        # 3. Create the two views for PointNeXt
        # VIEW A: The "Spatial Position" (Input XYZ)
        # We keep this as the centered coordinates for local grouping logic
        pos = pc_raw.copy() 

        # VIEW B: The "Input Features" (Standardized)
        # We scale by the constant 11.28 to provide a normalized feature vector
        # that preserves tree height proportions.
        x = pc_raw / 11.28 

        # 4. Return as Tensors
        return {
            "pos": torch.from_numpy(pos).float(),
            "x": torch.from_numpy(x).float(),
            "species_label": torch.tensor(
                row["label_idx"], dtype=torch.long
            ),  # Contiguous 0-15
            "structure_label": torch.tensor(float(row["h95"]), dtype=torch.float),
            "ecoregion": torch.tensor(
                row["ecoregion_idx"], dtype=torch.long
            ),  # Contiguous 0-10
        }


class PretextDataModule(LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.batch_size = config.get("batch_size", 32)
        self.num_workers = config.get("num_workers", 8)
        self.data_root = config["data_root"]

        # Paths to your static split files
        self.paths = {
            "train": config["train_csv"],  # "train_split.csv"
            "val": config["val_csv"],  # "val_split.csv"
            "test": config["test_csv"],  # "test_split.csv"
        }

        # Create a permanent mapping
        self.eco_to_idx = {name: i for i, name in enumerate(ONTARIO_ECOREGIONS)}

        self.species_to_idx = {label: i for i, label in enumerate(ONTARIO_SPECIES)}

        # Update config dynamically so the model knows the TRUE counts
        self.config["num_species"] = len(ONTARIO_SPECIES)  # 16
        self.config["num_ecoregions"] = len(ONTARIO_ECOREGIONS)  # 11

    def setup(self, stage=None):
        # Simply load the pre-calculated CSVs
        if stage == "fit" or stage is None:
            self.train_df = pd.read_csv(os.path.join(self.data_root, self.paths["train"]))
            self.val_df = pd.read_csv(os.path.join(self.data_root, self.paths["val"]))
        if stage == "test" or stage is None:
            self.test_df = pd.read_csv(os.path.join(self.data_root, self.paths["test"]))

        # 3. Safe Mapping Logic
        # We loop through possible attributes and only map if they exist
        for attr in ["train_df", "val_df", "test_df"]:
            if hasattr(self, attr):
                df = getattr(self, attr)
                if df is not None:
                    # Map Species Labels (e.g., 32 -> 15)
                    df["label_idx"] = df["label"].map(self.species_to_idx)
                    # Map Ecoregions (e.g., "6E" -> 10)
                    df["ecoregion_idx"] = df["ecoregion"].map(self.eco_to_idx)

                    # DROP rows that don't fit our allowed lists (Safety first!)
                    initial_len = len(df)
                    df.dropna(subset=["label_idx", "ecoregion_idx"], inplace=True)
                    df["label_idx"] = df["label_idx"].astype(int)
                    df["ecoregion_idx"] = df["ecoregion_idx"].astype(int)

                    if len(df) < initial_len:
                        print(
                            f"Dropped {initial_len - len(df)} plots due to unknown species/ecoregions in {attr}"
                        )

    def train_dataloader(self):
        ds = OntarioPretextDataset(
            self.train_df,
            self.data_root,
            eco_column="ecoregion_idx",
            transform=self.config.get("point_cloud_transform"),
            rotate=self.config.get("rotate", True),
        )
        return DataLoader(
            ds, batch_size=self.batch_size, shuffle=True, 
            num_workers=self.num_workers, pin_memory=True, drop_last=True
        )

    def val_dataloader(self):
        ds = OntarioPretextDataset(
            self.val_df,
            self.data_root,
            eco_column="ecoregion_idx"
        )
        return DataLoader(
            ds, batch_size=self.batch_size, shuffle=False, 
            num_workers=self.num_workers, pin_memory=True
        )

    def test_dataloader(self):
        ds = OntarioPretextDataset(
            self.test_df,
            self.data_root,
            eco_column="ecoregion_idx"
        )
        return DataLoader(
            ds, batch_size=self.batch_size, shuffle=False, 
            num_workers=self.num_workers
        )
