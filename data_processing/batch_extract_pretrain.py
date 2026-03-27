import os
import argparse
import numpy as np
import geopandas as gpd
import pdal
import json
import subprocess
import sys
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from pyproj import Transformer, CRS


# -------------------------
# Logging
# -------------------------
def log_message(message, log_file=None):
    print(message)
    sys.stdout.flush()
    if log_file:
        with open(log_file, "a") as f:
            f.write(message + "\n")


# -------------------------
# Read CRS + bounds from LAZ
# -------------------------
def get_tile_crs_and_bounds(laz_path):
    try:
        pipe = pdal.Pipeline(json.dumps([
            {"type": "readers.las", "filename": str(laz_path), "count": 1}
        ]))
        pipe.execute()

        meta = pipe.metadata
        if isinstance(meta, str):
            meta = json.loads(meta)

        reader = meta["metadata"]["readers.las"]

        bounds = {
            "minx": reader["minx"],
            "maxx": reader["maxx"],
            "miny": reader["miny"],
            "maxy": reader["maxy"],
        }

        wkt = reader["srs"]["compoundwkt"]
        crs = CRS.from_wkt(wkt)

        epsg = crs.to_epsg()

        if epsg is None:
            raise ValueError("CRS has no EPSG code")

        return crs, epsg, bounds

    except Exception as e:
        raise RuntimeError(f"Failed reading CRS from {laz_path}: {e}")


# -------------------------
# Build transformer safely
# -------------------------
def build_transformer(src_epsg, dst_crs):
    try:
        transformer = Transformer.from_crs(
            CRS.from_epsg(src_epsg),
            dst_crs,
            always_xy=True
        )
        return transformer
    except Exception as e:
        raise RuntimeError(f"Transformer creation failed: {e}")


# -------------------------
# Coordinate sanity check
# -------------------------
def is_valid_utm(x, y):
    return (0 < x < 1_000_000) and (0 < y < 10_000_000)


# -------------------------
# Process single plot
# -------------------------
def process_single_plot(laz_path, row, output_folder, transformer,
                        target_n=7168, debug_log=None):

    try:
        # Transform coordinates
        cx, cy = transformer.transform(row["x"], row["y"])

        # Validate
        if not is_valid_utm(cx, cy):
            return f"BAD_COORD_{cx:.1f}_{cy:.1f}"

        if debug_log:
            log_message(
                f"[COORD] 3978({row['x']:.1f},{row['y']:.1f}) -> TILE({cx:.1f},{cy:.1f})",
                debug_log
            )

        species_id = int(row["label"])

        pipeline = [
            {"type": "readers.las", "filename": str(laz_path)},
            {
                "type": "filters.crop",
                "point": f"POINT({cx:.3f} {cy:.3f})",
                "distance": 11.28
            },
            {"type": "filters.range", "limits": "Z(2:)"}
        ]

        pipe = pdal.Pipeline(json.dumps(pipeline))
        pipe.execute()

        if not pipe.arrays:
            return "EMPTY"

        pts = pipe.arrays[0]

        if len(pts) < target_n:
            return "DENSITY"

        data = np.vstack((
            pts["X"] - cx,
            pts["Y"] - cy,
            pts["Z"]
        )).T

        idx = np.random.choice(len(data), target_n, replace=False)

        out_path = (
            Path(output_folder)
            / str(species_id)
            / f"{species_id}_{int(row['x'])}_{int(row['y'])}.npy"
        )

        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_path, data[idx].astype(np.float32))

        return "SUCCESS"

    except Exception as e:
        return f"ERROR_{str(e)}"


# -------------------------
# Main
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_gpkg", type=str, required=True)
    parser.add_argument("--output_folder", type=str, required=True)
    parser.add_argument("--total_chunks", type=int, required=True)
    parser.add_argument("--chunk_idx", type=int, required=True)
    parser.add_argument("--num_workers", type=int, default=8)
    args = parser.parse_args()

    log_file = f"log_chunk_{args.chunk_idx}.txt"
    log_message(f"--- START CHUNK {args.chunk_idx} ---", log_file)

    gdf = gpd.read_file(args.input_gpkg, layer="sampling_plan_10k")

    # Ensure source CRS is 3978
    if gdf.crs is None:
        raise ValueError("Input GPKG has no CRS!")

    if gdf.crs.to_epsg() != 3978:
        log_message(f"[WARN] Reprojecting input to EPSG:3978", log_file)
        gdf = gdf.to_crs(epsg=3978)

    tiles = np.array_split(gdf["Tilename"].unique(), args.total_chunks)[args.chunk_idx]

    tmp_dir = Path(os.environ.get("SLURM_TMPDIR", "/tmp")) / f"batch_{args.chunk_idx}"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    stats = {"SUCCESS": 0, "EMPTY": 0, "DENSITY": 0, "BAD": 0, "ERROR": 0}

    for i, tile in enumerate(tiles):
        tile_df = gdf[gdf["Tilename"] == tile]

        url = tile_df["Download_H"].iloc[0]
        laz_path = tmp_dir / url.split("/")[-1]

        log_message(f"\n[TILE {i}] {tile}", log_file)

        subprocess.run(["wget", "-q", "-O", str(laz_path), url])

        if not laz_path.exists():
            log_message("[ERROR] Download failed", log_file)
            continue

        try:
            dst_crs, epsg, bounds = get_tile_crs_and_bounds(laz_path)

            log_message(
                f"[CRS] EPSG:{epsg} | Bounds X[{bounds['minx']:.0f}]",
                log_file
            )

            transformer = build_transformer(3978, dst_crs)

        except Exception as e:
            log_message(f"[ERROR] CRS issue: {e}", log_file)
            laz_path.unlink(missing_ok=True)
            continue

        # Parallel processing
        with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
            futures = [
                executor.submit(
                    process_single_plot,
                    laz_path,
                    row,
                    args.output_folder,
                    transformer,
                    7168,
                    log_file if j == 0 else None
                )
                for j, (_, row) in enumerate(tile_df.iterrows())
            ]

            for f in futures:
                res = f.result()

                if res == "SUCCESS":
                    stats["SUCCESS"] += 1
                elif res == "EMPTY":
                    stats["EMPTY"] += 1
                elif res == "DENSITY":
                    stats["DENSITY"] += 1
                elif res.startswith("BAD_COORD"):
                    stats["BAD"] += 1
                else:
                    stats["ERROR"] += 1

        laz_path.unlink(missing_ok=True)

        log_message(
            f"[PROGRESS] OK={stats['SUCCESS']} EMPTY={stats['EMPTY']} BAD={stats['BAD']}",
            log_file
        )

    log_message(f"\n--- DONE CHUNK {args.chunk_idx} ---", log_file)


if __name__ == "__main__":
    main()