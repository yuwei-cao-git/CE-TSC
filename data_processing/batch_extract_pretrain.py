import os
import argparse
import numpy as np
import pandas as pd
import geopandas as gpd
import pdal
import json
import subprocess
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def process_single_plot(laz_path, row, output_folder, target_n=7168):
    """
    Clips a 400m2 circle, filters points > 2m, calculates metrics,
    samples to 7168, and saves as .npy
    """
    center_x, center_y = row["x"], row["y"]
    species_id = int(row["label"])

    # 1. PDAL Pipeline for Canopy extraction (>2m)
    # We use height-normalized (HAG) data, so Z=height
    pipeline_json = [
        {"type": "readers.las", "filename": str(laz_path)},
        {
            "type": "filters.crop",
            "point": f"POINT({center_x} {center_y})",
            "distance": 11.28,
        },
        {"type": "filters.range", "limits": "Z(2:)"},
    ]

    # Pipeline for Total Returns (denominator for Canopy Cover)
    pipeline_raw_json = [
        {"type": "readers.las", "filename": str(laz_path)},
        {
            "type": "filters.crop",
            "point": f"POINT({center_x} {center_y})",
            "distance": 11.28,
        },
    ]

    try:
        # Get raw count for CC denominator
        pipe_raw = pdal.Pipeline(json.dumps(pipeline_raw_json))
        pipe_raw.execute()
        total_n = len(pipe_raw.arrays[0])
        if total_n == 0:
            return None

        # Get canopy points
        pipe_canopy = pdal.Pipeline(json.dumps(pipeline_json))
        pipe_canopy.execute()
        pts = pipe_canopy.arrays[0]
        canopy_n = len(pts)

        if canopy_n < target_n:
            return None  # Skip if too few points survive >2m filter

        # Metrics
        cc = (canopy_n / total_n) * 100
        ch = np.percentile(pts["Z"], 95)  # H95

        # Feature Prep: Center X/Y relative to plot, keep Z
        data = np.vstack((pts["X"] - center_x, pts["Y"] - center_y, pts["Z"])).T

        # Sample exactly target_n
        idx = np.random.choice(data.shape[0], target_n, replace=False)
        final_pts = data[idx].astype(np.float32)

        # Save Logic
        npy_name = f"{species_id}_{int(center_x)}_{int(center_y)}.npy"
        species_dir = Path(output_folder) / str(species_id)
        species_dir.mkdir(parents=True, exist_ok=True)
        npy_path = species_dir / npy_name

        np.save(npy_path, final_pts)

        return {
            "plot_id": npy_name,
            "label": species_id,
            "ecoregion": row.get("SITE_REGION_O", "unknown"),
            "canopy_cover": cc,
            "canopy_height": ch,
            "npy_path": str(npy_path),
        }
    except Exception as e:
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_gpkg", type=str, required=True)
    parser.add_argument("--output_folder", type=str, required=True)
    parser.add_argument("--total_chunks", type=int, required=True)
    parser.add_argument("--chunk_idx", type=int, required=True)
    parser.add_argument("--num_workers", type=int, default=16)
    args = parser.parse_args()

    # Load master plan and subset based on Tilename to ensure we only download each tile once
    gdf = gpd.read_file(args.input_gpkg, layer="sampling_plan_10k")
    unique_tiles = gdf["Tilename"].unique()
    chunk_tiles = np.array_split(unique_tiles, args.total_chunks)[args.chunk_idx]

    metadata = []
    tmp_dir = Path(os.environ.get("SLURM_TMPDIR", "/tmp"))

    for tile_name in tqdm(chunk_tiles, desc=f"Chunk {args.chunk_idx}"):
        tile_df = gdf[gdf["Tilename"] == tile_name]
        url = tile_df["Download_H"].iloc[0]
        local_laz = tmp_dir / f"{tile_name}.laz"

        # Download tile to local SSD
        dl = subprocess.run(["wget", "-q", "-O", str(local_laz), url])
        if dl.returncode != 0 or not local_laz.exists():
            continue

        # Extract plots from this tile
        with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
            futures = [
                executor.submit(process_single_plot, local_laz, row, args.output_folder)
                for _, row in tile_df.iterrows()
            ]
            for f in futures:
                res = f.result()
                if res:
                    metadata.append(res)

        # Cleanup large LAZ file immediately
        local_laz.unlink()

    # Save batch metadata for later merging
    pd.DataFrame(metadata).to_csv(f"meta_batch_{args.chunk_idx}.csv", index=False)


if __name__ == "__main__":
    main()
