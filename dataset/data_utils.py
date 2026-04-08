import numpy as np
import random

def center_xy_only(coords):
    """Centers X and Y but preserves Z (Height Above Ground)."""
    mean_xy = np.mean(coords[:, :2], axis=0)
    coords[:, :2] -= mean_xy
    return coords


def center_point_cloud(xyz):
    xyz_min = np.amin(xyz, axis=0, keepdims=True)
    xyz_max = np.amax(xyz, axis=0, keepdims=True)
    xyz_center = (xyz_min + xyz_max) / 2
    xyz_center[0][-1] = xyz_min[0][-1]
    xyz = xyz - xyz_center
    return xyz


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


def point_removal(coords, n, x=None):
    # Get list of ids
    idx = list(range(np.shape(coords)[0]))
    random.shuffle(idx)  # shuffle ids
    idx = np.random.choice(
        idx, n, replace=False
    )  # pick points randomly removing up to 10% of points

    # Remove random values
    aug_coords = coords[idx, :]  # remove coords
    if x is None:  # remove x
        aug_x = None
    else:
        aug_x = x[idx, :]

    return aug_coords, aug_x


def point_translate(coords, translate_range=0.1, x=None):
    translation = np.random.uniform(-translate_range, translate_range)
    coords[:, 0:3] += translation
    if x is None:  # remove x
        aug_x = None
    else:
        aug_x = x + translation
    return coords, aug_x


def random_scale(coords, lo=0.9, hi=1.1, x=None):
    scaler = np.random.uniform(lo, hi)
    aug_coords = coords * scaler
    if x is None:
        aug_x = None
    else:
        aug_x = x * scaler
    return aug_coords, aug_x


def random_noise(coords, n, dim=1, x=None):
    # Random standard deviation value
    random_noise_sd = np.random.uniform(0.01, 0.025)

    # Add/Subtract noise
    if np.random.uniform(0, 1) >= 0.5:  # 50% chance to add
        aug_coords = coords + np.random.normal(
            0, random_noise_sd, size=(np.shape(coords)[0], 3)
        )
        if x is None:
            aug_x = None
        else:
            aug_x = x + np.random.normal(
                0, random_noise_sd, size=(np.shape(x)[0], dim)
            )  # added [0] and dim
    else:  # 50% chance to subtract
        aug_coords = coords - np.random.normal(
            0, random_noise_sd, size=(np.shape(coords)[0], 3)
        )
        if x is None:
            aug_x = None
        else:
            aug_x = x - np.random.normal(
                0, random_noise_sd, size=(np.shape(x)[0], dim)
            )  # added [0] and dim

    # Randomly choose up to 10% of augmented noise points
    use_idx = np.random.choice(
        aug_coords.shape[0],
        n,
        replace=False,
    )
    aug_coords = aug_coords[use_idx, :]  # get random points
    aug_coords = np.append(coords, aug_coords, axis=0)  # add points
    if x is None:
        aug_x = None
    else:
        aug_x = aug_x[use_idx, :]  # get random point values
        aug_x = np.append(x, aug_x, axis=0)  # add random point values # ADDED axis=0

    return aug_coords, aug_x


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
    # Point Removal
    n = random.randint(round(len(xyz) * 0.9), len(xyz))
    aug_xyz, aug_feats = point_removal(xyz, n, x=pc_feat)
    aug_xyz, aug_feats = random_noise(aug_xyz, n=(len(xyz) - n), x=aug_feats)

    aug_xyz, aug_feats = random_scale(aug_xyz, x=aug_feats)
    aug_xyz, aug_feats = point_translate(aug_xyz, x=aug_feats)
    if rot:
        aug_xyz, aug_feats = rotate_z_only(aug_xyz, x=aug_feats)

    target = target

    return aug_xyz, aug_feats, target
