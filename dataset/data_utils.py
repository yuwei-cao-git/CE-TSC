import numpy as np
import torch


def center_xy_only(coords):
    """Centers X and Y but preserves Z (Height Above Ground)."""
    mean_xy = np.mean(coords[:, :2], axis=0)
    coords[:, :2] -= mean_xy
    return coords


def normalize_point_cloud(xyz):
    # Center and scale spatial coordinates
    centroid = np.mean(xyz, axis=0)
    xyz_centered = xyz - centroid
    max_distance = np.max(np.linalg.norm(xyz_centered, axis=1))
    xyz_normalized = xyz_centered / (max_distance + 1e-8)

    return xyz_normalized


def rotate_z_only(coords, feats=None):
    """
    Standard Forest LiDAR Augmentation: 
    Rotate only around the Z-axis (upright) to preserve gravity-based structure.
    """
    theta = np.random.uniform(0, 2 * np.pi)
    cos_val, sin_val = np.cos(theta), np.sin(theta)
    
    # Rotation matrix for Z-axis
    rot_mat = np.array([
        [cos_val, -sin_val, 0],
        [sin_val,  cos_val, 0],
        [0,        0,       1]
    ], dtype=np.float32)
    
    aug_coords = np.dot(coords, rot_mat)
    
    aug_feats = None
    if feats is not None:
        # Rotate features if they are also XYZ coordinates or Normals
        aug_feats = np.dot(feats, rot_mat)
        
    return aug_coords, aug_feats

def random_scale_isotropic(coords, feats=None, lo=0.95, hi=1.05):
    """
    Apply a single scalar to all axes to preserve tree aspect ratio (allometry).
    """
    scaler = np.random.uniform(lo, hi)
    aug_coords = coords * scaler
    
    aug_feats = None
    if feats is not None:
        aug_feats = feats * scaler
        
    return aug_coords, aug_feats

def jitter_points(coords, sigma=0.01, clip=0.05):
    """
    Simulate sensor noise by adding small Gaussian jitter.
    """
    noise = np.clip(sigma * np.random.randn(*coords.shape), -clip, clip).astype(np.float32)
    return coords + noise

def point_cloud_standardize(xyz, radius=11.28):
    """
    Isotropic Scaling for Deep Learning Stability.
    Maps X, Y to [-1, 1] while preserving vertical height proportions.
    """
    # Note: xyz is assumed to be centered at (0,0) in X,Y from the clipping script
    xyz_norm = xyz / radius
    return xyz_norm

def forest_pretext_transform(xyz, pc_feat=None, target=None, rot=True):
    """
    Master augmentation pipeline for Ontario Pre-training.
    """
    # 1. Random Rotation (Z-axis only)
    if rot:
        xyz, pc_feat = rotate_z_only(xyz, feats=pc_feat)
    
    # 2. Random Scaling (Isotropic)
    xyz, pc_feat = random_scale_isotropic(xyz, feats=pc_feat)
    
    # 3. Random Jitter (Sensor Noise)
    xyz = jitter_points(xyz)
    
    # 4. Local Translation (Optional: Simulate GPS offset, XY only)
    offset = np.random.uniform(-0.2, 0.2, size=(3,)).astype(np.float32)
    offset[2] = 0 # Lock Z
    xyz += offset
    
    return xyz, pc_feat, target
