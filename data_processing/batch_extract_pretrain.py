import os
import argparse
import numpy as np
import pandas as pd
import geopandas as gpd
import pdal
import json
import subprocess
import sys
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
        meta = json.loads(pipe.metadata)
        reader = meta["metadata"]["readers.las"]
        bounds = {"minx": reader["minx"], "maxx": reader["maxx"], 
                  "miny": reader["miny"], "maxy": reader["maxy"]}
        wkt = reader["srs"]["compoundwkt"]
        return CRS.from_wkt(wkt), bounds
    except Exception as e:
        raise RuntimeError(f"Failed reading CRS: {e}")

def build_transformer(src_epsg, dst_crs):
    return Transformer.from_crs(CRS.from_epsg(src_epsg), dst_crs, always_xy=True)

# -------------------------
# Process Single Plot
# -------------------------
def process_single_plot(laz_path, row, output_folder, transformer, target_n=8192):
    try:
        species_id = int(row["label"])
        eco_id = int(row.get("ecoregion", 0))
        rel_dir = str(species_id)
        file_name = f"{species_id}_{int(row['x'])}_{int(row['y'])}.npy"
        out_path = Path(output_folder) / rel_dir / file_name

        # --- PLOT LEVEL SKIP ---
        # If the file exists, we still need to return its metadata for the CSV
        if out_path.exists():
            # Quickly load to get H95 if we don't have it, or estimate
            # Since we need H95 for the CSV, and loading NPY is fast:
            data_existing = np.load(out_path)
            h95_val = np.percentile(data_existing[:, 2], 95)
            return {
                "status": "SKIPPED",
                "rel_path": f"{rel_dir}/{file_name}",
                "h95": h95_val,
                "eco": eco_id,
                "label": species_id,
                "tile": row["Tilename"]
            }

        # --- PDAL PROCESSING ---
        cx, cy = transformer.transform(row["x"], row["y"])
        pipeline = [
            {"type": "readers.las", "filename": str(laz_path)},
            {"type": "filters.crop", "point": f"POINT({cx:.3f} {cy:.3f})", "distance": 11.28},
            {"type": "filters.range", "limits": "Z(2:)"}
        ]
        
        pipe = pdal.Pipeline(json.dumps(pipeline))
        pipe.execute()

        if not pipe.arrays:
            return {"status": "EMPTY"}

        pts = pipe.arrays[0]
        if len(pts) < 7168:
            return {"status": "DENSITY"}

        # Calculate structural proxy (H95)
        h95_val = np.percentile(pts["Z"], 95)

        data = np.vstack((pts["X"] - cx, pts["Y"] - cy, pts["Z"])).T
        idx = np.random.choice(len(data), target_n, replace=False)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_path, data[idx].astype(np.float32))

        return {
            "status": "SUCCESS",
            "rel_path": f"{rel_dir}/{file_name}",
            "h95": h95_val,
            "eco": eco_id,
            "label": species_id,
            "tile": row["Tilename"]
        }

    except Exception as e:
        return {"status": f"ERROR: {str(e)}"}

# -------------------------
# Main
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_gpkg", type=str, required=True)
    parser.add_argument("--output_folder", type=str, required=True)
    parser.add_argument("--chunk_idx", type=int, required=True)
    parser.add_argument("--total_chunks", type=int, required=True)
    parser.add_argument("--num_workers", type=int, default=8)
    args = parser.parse_args()

    # Setup Logging
    log_dir = Path("./logs")
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / f"log_chunk_{args.chunk_idx}.txt"
    log_message(f"--- RESUMING CHUNK {args.chunk_idx} ---", log_file)

    # Load GPKG
    gdf = gpd.read_file(args.input_gpkg, layer="sampling_plan_10k")
    tiles = np.array_split(gdf["Tilename"].unique(), args.total_chunks)[args.chunk_idx]
    
    tmp_dir = Path(os.environ.get("SLURM_TMPDIR", "/tmp")) / f"chunk_{args.chunk_idx}"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    records = []
    stats = {"SUCCESS": 0, "SKIPPED": 0, "EMPTY": 0, "DENSITY": 0}

    for i, tile in enumerate(tiles):
        tile_df = gdf[gdf["Tilename"] == tile]
        
        # Check if all files in tile exist to skip download
        all_exist = True
        for _, r in tile_df.iterrows():
            if not (Path(args.output_folder) / str(int(r['label'])) / f"{int(r['label'])}_{int(r['x'])}_{int(r['y'])}.npy").exists():
                all_exist = False; break
        
        if all_exist:
            log_message(f"[TILE {i}] {tile} exists. Reading metadata...", log_file)
            # Parallel check to build CSV records without downloading
            with ThreadPoolExecutor(max_workers=args.num_workers) as ex:
                results = list(ex.map(lambda r: process_single_plot(None, r, args.output_folder, None), [row for _, row in tile_df.iterrows()]))
        else:
            # Download and process
            url = tile_df["Download_H"].iloc[0]
            laz_path = tmp_dir / url.split("/")[-1]
            log_message(f"[TILE {i}] Downloading {tile}...", log_file)
            subprocess.run(["wget", "-q", "-O", str(laz_path), url])
            
            try:
                dst_crs, _ = get_tile_crs_and_bounds(laz_path)
                transformer = build_transformer(3978, dst_crs)
                with ThreadPoolExecutor(max_workers=args.num_workers) as ex:
                    results = [ex.submit(process_single_plot, laz_path, row, args.output_folder, transformer) for _, row in tile_df.iterrows()]
                    results = [r.result() for r in results]
            except Exception as e:
                log_message(f"Error on {tile}: {e}", log_file); continue
            finally:
                laz_path.unlink(missing_ok=True)

        # Update metadata records
        for res in results:
            if res["status"] in ["SUCCESS", "SKIPPED"]:
                records.append({
                    "relative_path": res["rel_path"],
                    "label": res["label"],
                    "h95": res["h95"],
                    "ecoregion": res["eco"],
                    "tilename": res["tile"]
                })
                stats[res["status"]] += 1
            else:
                stats[res["status"] if res["status"] in stats else "EMPTY"] += 1

    # Save CSV for this chunk
    pd.DataFrame(records).to_csv(f"meta_batch_{args.chunk_idx}.csv", index=False)
    log_message(f"Chunk {args.chunk_idx} Done. OK: {stats['SUCCESS']} Resumed: {stats['SKIPPED']}", log_file)

if __name__ == "__main__":
    main()