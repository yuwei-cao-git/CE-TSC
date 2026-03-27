import pandas as pd
import numpy as np
import os

import torch
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule

from data_utils import pointCloudTransform, normalize_point_cloud

class OntarioPretextDataset(Dataset):
    def __init__(
        self,
        manifest_df,
        data_root,
        transform=None,
        rotate=False,
        estimate_normals=False
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
        self.rotate = rotate
        self.estimate_normals = estimate_normals

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # 1. Load raw extracted points (centered XY, absolute Z meters)
        # Shape: (N, 3) -> [X_rel, Y_rel, Z_hag]
        pc_raw = np.load(os.path.join(self.data_root, row['relative_path'])).astype(np.float32)
        
        # 2. Augmentation (Rotation/Scaling/Jitter)
        # Apply these to the 'raw' meter-scale points first
        if self.transform:
            pc_raw, _, _ = self.transform(pc_raw, pc_feat=None, target=None)

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
            "pos": torch.from_numpy(pos).float(), # Used for Ball Query / Grouping
            "x": torch.from_numpy(x).float(),     # Used as the initial point feature
            "species_label": torch.tensor(int(row['label']), dtype=torch.long),
            "structure_label": torch.tensor(float(row['h95']), dtype=torch.float),
            "ecoregion": torch.tensor(int(row['ecoregion']), dtype=torch.long)
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
            "train": config["train_split_path"], # "train_split.csv"
            "val": config["val_split_path"],     # "val_split.csv"
            "test": config["test_split_path"]    # "test_split.csv"
        }

    def setup(self, stage=None):
        # Simply load the pre-calculated CSVs
        if stage == "fit" or stage is None:
            self.train_df = pd.read_csv(self.paths["train"])
            self.val_df = pd.read_csv(self.paths["val"])
        if stage == "test" or stage is None:
            self.test_df = pd.read_csv(self.paths["test"])

    def train_dataloader(self):
        ds = OntarioPretextDataset(
            self.train_df, 
            self.data_root, 
            transform=self.config.get("point_cloud_transform"),
            rotate=self.config.get("rotate", True)
        )
        return DataLoader(
            ds, batch_size=self.batch_size, shuffle=True, 
            num_workers=self.num_workers, pin_memory=True, drop_last=True
        )

    def val_dataloader(self):
        ds = OntarioPretextDataset(self.val_df, self.data_root)
        return DataLoader(
            ds, batch_size=self.batch_size, shuffle=False, 
            num_workers=self.num_workers, pin_memory=True
        )

    def test_dataloader(self):
        ds = OntarioPretextDataset(self.test_df, self.data_root)
        return DataLoader(
            ds, batch_size=self.batch_size, shuffle=False, 
            num_workers=self.num_workers
        )
