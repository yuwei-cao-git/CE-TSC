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
    center_x, center_y = row["x"], row["y"]
    species_id = int(row["label"])

    # Define PDAL Pipeline
    pipeline_json = [
        {"type": "readers.las", "filename": str(laz_path)},
        {
            "type": "filters.crop",
            "point": f"POINT({center_x} {center_y})",
            "distance": 11.28,
        },
        {"type": "filters.range", "limits": "Z(2:)"},
    ]

    pipeline_raw_json = [
        {"type": "readers.las", "filename": str(laz_path)},
        {
            "type": "filters.crop",
            "point": f"POINT({center_x} {center_y})",
            "distance": 11.28,
        },
    ]

    try:
        # Denominator for CC
        pipe_raw = pdal.Pipeline(json.dumps(pipeline_raw_json))
        pipe_raw.execute()
        total_n = len(pipe_raw.arrays[0])
        if total_n == 0:
            return None

        # Canopy points
        pipe_canopy = pdal.Pipeline(json.dumps(pipeline_json))
        pipe_canopy.execute()
        pts = pipe_canopy.arrays[0]
        canopy_n = len(pts)

        if canopy_n < target_n:
            return None

        # Compute metrics
        cc = (canopy_n / total_n) * 100
        ch = np.percentile(pts["Z"], 95)

        # Process coordinates
        data = np.vstack((pts["X"] - center_x, pts["Y"] - center_y, pts["Z"])).T
        idx = np.random.choice(data.shape[0], target_n, replace=False)
        final_pts = data[idx].astype(np.float32)

        # SAVE IMMEDIATELY TO SCRATCH
        npy_name = f"{species_id}_{int(center_x)}_{int(center_y)}.npy"
        species_dir = Path(output_folder) / str(species_id)
        species_dir.mkdir(parents=True, exist_ok=True)
        npy_path = species_dir / npy_name

        np.save(str(npy_path), final_pts)

        return {
            "plot_id": npy_name,
            "label": species_id,
            "ecoregion": row.get("SITE_REGION_O", "unknown"),
            "canopy_cover": cc,
            "canopy_height": ch,
            "npy_path": str(npy_path),
        }
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_gpkg", type=str, required=True)
    parser.add_argument("--output_folder", type=str, required=True)
    parser.add_argument("--total_chunks", type=int, required=True)
    parser.add_argument("--chunk_idx", type=int, required=True)
    parser.add_argument("--num_workers", type=int, default=16)
    args = parser.parse_args()

    # Ensure output_folder is absolute
    args.output_folder = os.path.abspath(args.output_folder)

    gdf = gpd.read_file(args.input_gpkg, layer="sampling_plan_10k")
    unique_tiles = gdf["Tilename"].unique()
    chunk_tiles = np.array_split(unique_tiles, args.total_chunks)[args.chunk_idx]

    metadata = []

    # Check SLURM_TMPDIR
    tmp_env = os.environ.get("SLURM_TMPDIR")
    if not tmp_env:
        print("Warning: SLURM_TMPDIR not found, using /tmp")
        tmp_env = "/tmp"

    tmp_dir = Path(tmp_env)

    for tile_name in tqdm(chunk_tiles, desc=f"Chunk {args.chunk_idx}"):
        tile_df = gdf[gdf["Tilename"] == tile_name]
        url = tile_df["Download_H"].iloc[0]
        local_laz = tmp_dir / f"{tile_name}.laz"

        # Explicit Download Check
        if not local_laz.exists():
            subprocess.run(["wget", "-q", "-O", str(local_laz), url], check=False)

        if not local_laz.exists() or local_laz.stat().st_size == 0:
            print(f"Failed to download tile: {tile_name}")
            continue

        with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
            futures = [
                executor.submit(process_single_plot, local_laz, row, args.output_folder)
                for _, row in tile_df.iterrows()
            ]
            for f in futures:
                res = f.result()
                if res:
                    metadata.append(res)

        # Cleanup current tile to free up node space
        if local_laz.exists():
            local_laz.unlink()

    # Save metadata for this chunk
    if metadata:
        pd.DataFrame(metadata).to_csv(f"meta_batch_{args.chunk_idx}.csv", index=False)


if __name__ == "__main__":
    main()
