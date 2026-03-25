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
from pyproj import Transformer


def get_native_epsg(laz_path):
    """Detects the EPSG code of the LAS/LAZ file using PDAL metadata."""
    try:
        pipeline = pdal.Pipeline(
            json.dumps(
                [
                    {"type": "readers.las", "filename": str(laz_path)},
                    {"type": "filters.stats", "count": 1},
                ]
            )
        )
        pipeline.execute()
        metadata = json.loads(pipeline.metadata)
        # Try to find the EPSG in the reader metadata
        srs = metadata["metadata"]["readers.las"]["srs"]["compoundwkt"]
        # Look for the EPSG:XXXX string
        if "EPSG" in srs:
            import re

            match = re.search(r'EPSG",(\d+)', srs)
            if match:
                return f"EPSG:{match.group(1)}"
        return "EPSG:2959"  # Fallback to UTM 18N if detection fails
    except:
        return "EPSG:2959"


def process_single_plot(laz_path, row, output_folder, transformer, target_n=7168):
    """Uses the provided transformer to move NTEMS 3978 to Tile's Native CRS."""
    # Transform center from 3978 to Tile CRS (e.g., 2959)
    center_x_tile, center_y_tile = transformer.transform(row["x"], row["y"])
    species_id = int(row["label"])

    p_canopy = [
        {"type": "readers.las", "filename": str(laz_path)},
        {
            "type": "filters.crop",
            "point": f"POINT({center_x_tile} {center_y_tile})",
            "distance": 11.28,
        },
        {"type": "filters.range", "limits": "Z(2:)"},
    ]
    p_raw = [
        {"type": "readers.las", "filename": str(laz_path)},
        {
            "type": "filters.crop",
            "point": f"POINT({center_x_tile} {center_y_tile})",
            "distance": 11.28,
        },
    ]

    try:
        # Get counts for CC
        pipe_r = pdal.Pipeline(json.dumps(p_raw))
        pipe_r.execute()
        total_n = len(pipe_raw.arrays[0])
        if total_n == 0:
            return None

        # Get Points
        pipe_c = pdal.Pipeline(json.dumps(p_canopy))
        pipe_c.execute()
        pts = pipe_canopy.arrays[0]
        canopy_n = len(pts)

        if canopy_n < target_n:
            return None

        # Feature Prep (Center X/Y relative to plot)
        data = np.vstack(
            (pts["X"] - center_x_tile, pts["Y"] - center_y_tile, pts["Z"])
        ).T
        idx = np.random.choice(data.shape[0], target_n, replace=False)
        final_pts = data[idx].astype(np.float32)

        # Output
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
        }
    except:
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_gpkg", type=str, required=True)
    parser.add_argument("--output_folder", type=str, required=True)
    parser.add_argument("--total_chunks", type=int, required=True)
    parser.add_argument("--chunk_idx", type=int, required=True)
    parser.add_argument("--num_workers", type=int, default=16)
    args = parser.parse_args()

    gdf = gpd.read_file(args.input_gpkg, layer="sampling_plan_10k")
    unique_tiles = gdf["Tilename"].unique()
    chunk_tiles = np.array_split(unique_tiles, args.total_chunks)[args.chunk_idx]

    tmp_dir = Path(os.environ.get("SLURM_TMPDIR", "/tmp"))
    metadata = []

    for tile_name in tqdm(chunk_tiles):
        tile_df = gdf[gdf["Tilename"] == tile_name]
        local_laz = tmp_dir / f"{tile_name}.laz"

        # Download
        subprocess.run(
            ["wget", "-q", "-O", str(local_laz), tile_df["Download_H"].iloc[0]]
        )
        if not local_laz.exists():
            continue

        # --- DYNAMIC CRS DETECTION ---
        native_epsg = get_native_epsg(local_laz)
        # Create a specific transformer for this tile
        tile_transformer = Transformer.from_crs(
            "EPSG:3978", native_epsg, always_xy=True
        )
        # -----------------------------

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

        local_laz.unlink()

    if metadata:
        pd.DataFrame(metadata).to_csv(f"meta_batch_{args.chunk_idx}.csv", index=False)


if __name__ == "__main__":
    main()
