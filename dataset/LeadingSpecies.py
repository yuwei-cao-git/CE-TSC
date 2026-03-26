import torch
from torch.utils.data import Dataset
import numpy as np
import os
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from os.path import join
from .augment import pointCloudTransform, normalize_point_cloud, center_point_cloud
import open3d as o3d

class LeadingSpeciesDataset(Dataset):
    def __init__(
        self,
        superpixel_files,
        rotate=None,
        pc_normal=None,
        point_cloud_transform=None
    ):
        self.superpixel_files = superpixel_files
        self.point_cloud_transform = point_cloud_transform
        self.rotate = rotate
        self.normal = pc_normal

    def __len__(self):
        return len(self.superpixel_files)

    def __getitem__(self, idx):
        data = np.load(self.superpixel_files[idx], allow_pickle=True)
        # Load data from the .npz file
        coords = data["point_cloud"]  # Shape: (7168, 3)
        label = data["label"]  # Shape: (num_classes,)
        centred_coords = center_point_cloud(coords)
        if self.normal:
            # Convert numpy array to Open3D PointCloud
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(centred_coords)

            # Estimate normals (change radius/knn depending on density)
            pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=16))
            feats = np.asarray(pcd.normals)  # Shape: (N, 3)
        else:
            norm_coords = normalize_point_cloud(coords)
            feats = norm_coords
        
        # Apply point cloud transforms if any
        if self.point_cloud_transform:
            centred_coords, feats, label = pointCloudTransform(
                centred_coords, pc_feat=feats, target=label, rot=self.rotate
            )

        # After applying transforms
        feats = torch.from_numpy(feats).float()  # Shape: (7168, 3)
        centred_coords = torch.from_numpy(centred_coords).float()  # Shape: (7168, 3)
        label = torch.from_numpy(label).float()  # Shape: (num_classes,)

        sample = {
            "point_cloud": centred_coords,
            "pc_feat": feats,
            "label": label
        }
        return sample


class LSDataModule(LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.dataset = config["dataset"]
        self.test_dataset = config["test_dataset"]
        self.batch_size = config["batch_size"]
        self.num_workers = config["gpus"]*2
        self.point_cloud_transform = config["point_cloud_transform"]
        self.aug_rotate = config["rotate"]
        self.aug_pc_norm = config["pc_normal"]
        self.data_dirs = {
            "train": join(
                config["data_dir"], "tile_128", "train", f"{config['dataset']}"
            ),
            "val": join(config["data_dir"], "tile_128", "val", f"{config['dataset']}"),
            "test": join(
                config["test_data_dir"], "tile_128", "test", f"{config['test_dataset']}"
            ),
        }

    def setup(self, stage=None):
        # Create datasets for train, validation, and test
        self.datasets = {}
        for split in ["train", "val", "test"]:
            data_dir = self.data_dirs[split]
            superpixel_files = [
                os.path.join(data_dir, f)
                for f in os.listdir(data_dir)
                if f.endswith(".npz")
            ]

            self.datasets[split] = LeadingSpeciesDataset(
                superpixel_files,
                rotate=None,
                pc_normal=self.aug_pc_norm,
                point_cloud_transform=None
            )
            if split == "train":
                if self.point_cloud_transform:
                    aug_pc_dataset = LeadingSpeciesDataset(
                        superpixel_files,
                        rotate=self.aug_rotate,
                        pc_normal=self.aug_pc_norm,
                        point_cloud_transform=self.point_cloud_transform
                    )
                    self.datasets["train"] = torch.utils.data.ConcatDataset(
                        [self.datasets["train"], aug_pc_dataset]
                    )

    def train_dataloader(self):
        return DataLoader(
            self.datasets["train"],
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.datasets["val"],
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.datasets["test"],
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=False
        )
