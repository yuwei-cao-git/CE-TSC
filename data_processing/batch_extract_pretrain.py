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
    """Detects the EPSG code from LAZ/COPC metadata (Fixed for PDAL 2.x)."""
    try:
        pipeline = pdal.Pipeline(
            json.dumps([{"type": "readers.las", "filename": str(laz_path), "count": 1}])
        )
        pipeline.execute()

        # PDAL behavior varies: some versions return dict, some return JSON string
        meta = pipeline.metadata
        if isinstance(meta, str):
            meta = json.loads(meta)

        srs = meta["metadata"]["readers.las"]["srs"]["compoundwkt"]
        match = re.search(r'EPSG",(\d+)', srs)
        return f"EPSG:{match.group(1)}" if match else "EPSG:2959"
    except Exception as e:
        print(
            f"CRS Detection Warning for {laz_path.name}: {e}. Falling back to EPSG:2959."
        )
        return "EPSG:2959"


def process_single_plot(laz_path, row, output_folder, transformer, target_n=7168):
    """Clips and samples a plot. Saves immediately to Scratch."""
    try:
        center_x_tile, center_y_tile = transformer.transform(row["x"], row["y"])
        species_id = int(row["label"])

        # PDAL Pipelines
        p_crop = {
            "type": "filters.crop",
            "point": f"POINT({center_x_tile} {center_y_tile})",
            "distance": 11.28,
        }

        # Pipeline 1: Raw for CC
        pipe_raw = pdal.Pipeline(
            json.dumps([{"type": "readers.las", "filename": str(laz_path)}, p_crop])
        )
        pipe_raw.execute()
        total_n = len(pipe_raw.arrays[0])
        if total_n == 0:
            return None

        # Pipeline 2: Canopy for Points
        pipe_canopy = pdal.Pipeline(
            json.dumps(
                [
                    {"type": "readers.las", "filename": str(laz_path)},
                    p_crop,
                    {"type": "filters.range", "limits": "Z(2:)"},
                ]
            )
        )
        pipe_canopy.execute()
        pts = pipe_canopy.arrays[0]
        canopy_n = len(pts)

        if canopy_n < target_n:
            return None

        # Metrics and Sampling
        data = np.vstack(
            (pts["X"] - center_x_tile, pts["Y"] - center_y_tile, pts["Z"])
        ).T
        idx = np.random.choice(data.shape[0], target_n, replace=False)
        final_pts = data[idx].astype(np.float32)

        # Output logic
        npy_name = f"{species_id}_{int(row['x'])}_{int(row['y'])}.npy"
        species_dir = Path(output_folder) / str(species_id)
        species_dir.mkdir(parents=True, exist_ok=True)
        npy_path = species_dir / npy_name

        np.save(str(npy_path), final_pts)

        return {
            "plot_id": npy_name,
            "label": species_id,
            "ecoregion": row.get("SITE_REGION_O", "unknown"),
            "canopy_cover": (canopy_n / total_n) * 100,
            "canopy_height": np.percentile(pts["Z"], 95),
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
    gdf = gpd.read_file(args.input_gpkg, layer="sampling_plan_10k")
    unique_tiles = gdf["Tilename"].unique()
    chunk_tiles = np.array_split(unique_tiles, args.total_chunks)[args.chunk_idx]

    # Securely setup tmp_dir
    tmp_base = os.environ.get("SLURM_TMPDIR", "/tmp")
    tmp_dir = Path(tmp_base) / f"job_chunk_{args.chunk_idx}"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    metadata = []

    for tile_name in tqdm(chunk_tiles):
        tile_df = gdf[gdf["Tilename"] == tile_name]
        url = tile_df["Download_H"].iloc[0]

        # Use exact name from URL
        local_laz = tmp_dir / url.split("/")[-1]

        # Download with verification
        if not local_laz.exists():
            subprocess.run(["wget", "-q", "-O", str(local_laz), url], check=False)

        if not local_laz.exists() or local_laz.stat().st_size < 1000:
            continue

        # Setup Transformer for this specific tile
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

        # Cleanup
        if local_laz.exists():
            local_laz.unlink()

    # Save metadata chunk
    if metadata:
        pd.DataFrame(metadata).to_csv(f"meta_batch_{args.chunk_idx}.csv", index=False)


if __name__ == "__main__":
    main()
