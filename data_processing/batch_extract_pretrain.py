import os
import argparse
import numpy as np
import pandas as pd
import geopandas as gpd
import pdal
import json
import subprocess
import re
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from pyproj import Transformer


def get_native_epsg(laz_path):
    """Detects the EPSG code from the LAZ/COPC metadata."""
    try:
        # We only need to read the metadata header
        pipeline = pdal.Pipeline(
            json.dumps([{"type": "readers.las", "filename": str(laz_path), "count": 1}])
        )
        pipeline.execute()
        metadata = json.loads(pipeline.metadata)
        srs = metadata["metadata"]["readers.las"]["srs"]["compoundwkt"]

        # Search for EPSG code
        match = re.search(r'EPSG",(\d+)', srs)
        if match:
            return f"EPSG:{match.group(1)}"
        return "EPSG:2959"  # Default fallback for Ontario SPL
    except Exception as e:
        print(f"Metadata error on {laz_path.name}: {e}")
        return "EPSG:2959"


def process_single_plot(laz_path, row, output_folder, transformer, target_n=7168):
    """Clips, filters, samples, and saves a single plot."""
    # NTEMS (3978) -> Tile Native CRS (e.g., 2959)
    center_x_tile, center_y_tile = transformer.transform(row["x"], row["y"])
    species_id = int(row["label"])

    p_raw = [
        {"type": "readers.las", "filename": str(laz_path)},
        {
            "type": "filters.crop",
            "point": f"POINT({center_x_tile} {center_y_tile})",
            "distance": 11.28,
        },
    ]
    p_canopy = [
        {"type": "readers.las", "filename": str(laz_path)},
        {
            "type": "filters.crop",
            "point": f"POINT({center_x_tile} {center_y_tile})",
            "distance": 11.28,
        },
        {"type": "filters.range", "limits": "Z(2:)"},
    ]

    try:
        # Get raw count for CC denominator
        pipe_r = pdal.Pipeline(json.dumps(p_raw))
        pipe_r.execute()
        total_n = len(pipe_r.arrays[0])
        if total_n == 0:
            return None

        # Get canopy points
        pipe_c = pdal.Pipeline(json.dumps(p_canopy))
        pipe_c.execute()
        pts = pipe_c.arrays[0]
        canopy_n = len(pts)

        if canopy_n < target_n:
            return None

        # Metrics
        cc = (canopy_n / total_n) * 100
        ch = np.percentile(pts["Z"], 95)

        # Tensor: Center relative to plot center, keep Z as HAG
        data = np.vstack(
            (pts["X"] - center_x_tile, pts["Y"] - center_y_tile, pts["Z"])
        ).T

        # Fixed point sampling
        idx = np.random.choice(data.shape[0], target_n, replace=False)
        final_pts = data[idx].astype(np.float32)

        # Immediate Save to Scratch
        npy_name = f"{species_id}_{int(row['x'])}_{int(row['y'])}.npy"
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

    args.output_folder = os.path.abspath(args.output_folder)

    # Load Sampling Plan
    gdf = gpd.read_file(args.input_gpkg, layer="sampling_plan_10k")
    unique_tiles = gdf["Tilename"].unique()
    chunk_tiles = np.array_split(unique_tiles, args.total_chunks)[args.chunk_idx]

    # Inside main()
    tmp_env = os.environ.get("SLURM_TMPDIR")
    if tmp_env:
        # Create a specific 'downloads' folder inside the Slurm temp space
        tmp_dir = Path(tmp_env) / "working"
        tmp_dir.mkdir(parents=True, exist_ok=True)
    else:
        tmp_dir = Path("./local_test_dir")
        tmp_dir.mkdir(exist_ok=True)
    metadata = []

    for tile_name in tqdm(chunk_tiles, desc=f"Batch {args.chunk_idx}"):
        tile_df = gdf[gdf["Tilename"] == tile_name]
        url = tile_df["Download_H"].iloc[0]

        # DYNAMIC FILENAME LOGIC: Extract actual filename from the URL
        remote_filename = url.split("/")[-1]
        local_laz = tmp_dir / remote_filename

        # Download Check
        if not local_laz.exists():
            subprocess.run(["wget", "-O", str(local_laz), url], check=False)

        if not local_laz.exists() or local_laz.stat().st_size == 0:
            print(f"ERROR: Could not download {url}")
            continue

        # DYNAMIC CRS DETECTION
        native_epsg = get_native_epsg(local_laz)
        tile_transformer = Transformer.from_crs(
            "EPSG:3978", native_epsg, always_xy=True
        )

        with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
            futures = [
                executor.submit(
                    process_single_plot,
                    local_laz,
                    row,
                    args.output_folder,
                    tile_transformer,
                )
                for _, row in tile_df.iterrows()
            ]
            for f in futures:
                res = f.result()
                if res:
                    metadata.append(res)

        # Instant Cleanup of the large tile
        if local_laz.exists():
            local_laz.unlink()

    # Save metadata for this batch
    if metadata:
        pd.DataFrame(metadata).to_csv(f"meta_batch_{args.chunk_idx}.csv", index=False)


if __name__ == "__main__":
    main()
