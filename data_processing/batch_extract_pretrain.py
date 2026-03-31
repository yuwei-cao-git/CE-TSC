import os
import argparse
import numpy as np
import pandas as pd
import geopandas as gpd
import pdal
import json
import subprocess
import sys
import gc
import glob
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from pyproj import Transformer, CRS
import re

# -------------------------
# Utilities
# -------------------------
def log_message(message, log_file=None):
    print(message)
    sys.stdout.flush()
    if log_file:
        with open(log_file, "a") as f:
            f.write(message + "\n")

ZONE_TO_EPSG = {
    15: 3159,
    16: 3160,
    17: 2958,
    18: 2959,
}


def infer_crs_from_tile_name(tile_name: str):
    """
    Extract UTM zone from tile name like:
    1kmZ153430547302022L
    1kmZ183260504102022L
    """
    m = re.search(r"Z(\d{2})", tile_name)
    if not m:
        raise ValueError(f"Cannot infer zone from tile name: {tile_name}")

    zone = int(m.group(1))

    if zone not in ZONE_TO_EPSG:
        raise ValueError(f"Unsupported UTM zone {zone} in {tile_name}")

    epsg = ZONE_TO_EPSG[zone]
    return CRS.from_epsg(epsg), epsg


def get_tile_crs(laz_path):
    try:
        pipe = pdal.Pipeline(
            json.dumps([{"type": "readers.las", "filename": str(laz_path), "count": 1}])
        )
        pipe.execute()

        meta = pipe.metadata
        if isinstance(meta, str):
            meta = json.loads(meta)

        reader = meta["metadata"]["readers.las"]

        # -------------------------
        # Priority 1: src.compoundwkt
        # -------------------------
        try:
            wkt = reader["src"]["compoundwkt"]

            if isinstance(wkt, str) and wkt.strip():
                crs = CRS.from_wkt(wkt)

                if crs.is_compound:
                    crs = crs.sub_crs_list[0]

                return crs

        except Exception:
            pass

        # -------------------------
        # Priority 2: srs
        # -------------------------
        srs_info = reader.get("srs", None)

        if isinstance(srs_info, dict):
            wkt = srs_info.get("compoundwkt", "") or srs_info.get("wkt", "")

            if wkt:
                try:
                    crs = CRS.from_wkt(wkt)

                    if crs.is_compound:
                        crs = crs.sub_crs_list[0]

                    return crs
                except Exception:
                    pass

        elif isinstance(srs_info, str):
            if srs_info.strip():
                try:
                    crs = CRS.from_wkt(srs_info)

                    if crs.is_compound:
                        crs = crs.sub_crs_list[0]

                    return crs
                except Exception:
                    pass

        # -------------------------
        # Final fallback: tile name
        # -------------------------
        print(f"[WARN] Using tile-name fallback for {laz_path.name}")
        return infer_crs_from_tile_name(laz_path.name)

    except Exception as e:
        raise RuntimeError(f"Failed reading CRS: {e}")


# -------------------------
# Worker Function
# -------------------------
def process_single_plot(laz_path, row, output_folder, transformer, target_n=8192):
    """Processes a single plot: Crop -> Filter -> H95 -> Save."""
    try:
        species_id = int(row["label"])
        eco_id = str(row.get("SITE_REG_O", 0))
        rel_dir = str(species_id)
        file_name = f"{species_id}_{int(row['x'])}_{int(row['y'])}.npy"
        out_path = Path(output_folder) / rel_dir / file_name

        # 1. PDAL Pipeline
        cx, cy = transformer.transform(row["x"], row["y"])
        pipeline = [
            {"type": "readers.las", "filename": str(laz_path)},
            {
                "type": "filters.crop",
                "point": f"POINT({cx:.3f} {cy:.3f})",
                "distance": 11.28,
            },
            {"type": "filters.range", "limits": "Z(2:)"},
        ]

        pipe = pdal.Pipeline(json.dumps(pipeline))
        pipe.execute()

        if not pipe.arrays:
            return None
        pts = pipe.arrays[0]
        if len(pts) < 7168:
            return None

        # 2. Extract structural proxy (H95)
        h95_val = float(np.percentile(pts["Z"], 95))

        # 3. Format and Save
        data = np.vstack((pts["X"] - cx, pts["Y"] - cy, pts["Z"])).T
        idx = np.random.choice(len(data), target_n, replace=False)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_path, data[idx].astype(np.float32))

        return {
            "relative_path": f"{rel_dir}/{file_name}",
            "label": species_id,
            "h95": h95_val,
            "ecoregion": eco_id,
            "tilename": row["Tilename"],
        }
    except Exception:
        return None


# -------------------------
# Main Execution
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_gpkg", type=str, required=True)
    parser.add_argument("--output_folder", type=str, required=True)
    parser.add_argument("--chunk_idx", type=int, required=True)
    parser.add_argument("--total_chunks", type=int, required=True)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument(
        "--merge_only", action="store_true", help="Only merge existing CSVs"
    )
    args = parser.parse_args()

    # --- MERGE LOGIC ---
    if args.merge_only:
        print("Merging all meta_batch_*.csv files...")
        all_csvs = glob.glob("./logs/meta_batch_*.csv")
        if not all_csvs:
            print("No CSV files found to merge.")
            return
        combined_df = pd.concat([pd.read_csv(f) for f in all_csvs])
        combined_df.to_csv("../dataset/training_master_list.csv", index=False)
        print(f"Success! Final manifest saved with {len(combined_df)} records.")
        return

    # --- EXTRACTION LOGIC ---
    log_dir = Path("./logs")
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / f"log_chunk_{args.chunk_idx}.txt"

    # Load and Split Work
    gdf = gpd.read_file(args.input_gpkg, layer="sampling_plan_10k")
    all_tiles = gdf["Tilename"].unique()
    tiles = np.array_split(all_tiles, args.total_chunks)[args.chunk_idx]

    tmp_dir = Path(os.environ.get("SLURM_TMPDIR", "/tmp")) / f"chunk_{args.chunk_idx}"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    records = []

    for i, tile in enumerate(tiles):
        tile_df = gdf[gdf["Tilename"] == tile]

        # 1. SMART SKIP: Identify what needs processing vs what is on disk
        needed_rows = []
        for _, r in tile_df.iterrows():
            p = (
                Path(args.output_folder)
                / str(int(r["label"]))
                / f"{int(r['label'])}_{int(r['x'])}_{int(r['y'])}.npy"
            )
            if p.exists():
                # Rapidly pull metadata from existing file to populate CSV
                try:
                    data_existing = np.load(p)
                    records.append(
                        {
                            "relative_path": f"{int(r['label'])}/{p.name}",
                            "label": int(r["label"]),
                            "h95": float(np.percentile(data_existing[:, 2], 95)),
                            "ecoregion": str(r.get("SITE_REG_O", 0)),
                            "tilename": tile,
                        }
                    )
                except:
                    pass  # Corrupt file, will re-process if laz is downloaded
            else:
                needed_rows.append(r)

        if not needed_rows:
            log_message(f"[TILE {i}/{len(tiles)}] {tile} -> Fully Resumed.", log_file)
            continue

        # 2. DOWNLOAD & PROCESS
        url = tile_df["Download_H"].iloc[0]
        laz_path = tmp_dir / url.split("/")[-1]
        log_message(
            f"[TILE {i}/{len(tiles)}] {tile} -> Downloading {len(needed_rows)} plots...",
            log_file,
        )
        subprocess.run(["wget", "-q", "-O", str(laz_path), url])

        if laz_path.exists():
            try:
                dst_crs = get_tile_crs(laz_path)
                trans = Transformer.from_crs(
                    CRS.from_epsg(3978), dst_crs, always_xy=True
                )

                with ThreadPoolExecutor(max_workers=args.num_workers) as ex:
                    futures = [
                        ex.submit(
                            process_single_plot, laz_path, r, args.output_folder, trans
                        )
                        for r in needed_rows
                    ]
                    for f in futures:
                        res = f.result()
                        if res:
                            records.append(res)
            except Exception as e:
                log_message(f"Error on {tile}: {e}", log_file)
            finally:
                laz_path.unlink(missing_ok=True)
                gc.collect()

    # 3. SAVE CHUNK CSV
    out_csv = f"./logs/meta_batch_{args.chunk_idx}.csv"
    pd.DataFrame(records).to_csv(out_csv, index=False)
    log_message(
        f"--- Chunk {args.chunk_idx} Complete. Total Records: {len(records)} ---",
        log_file,
    )


if __name__ == "__main__":
    main()

# python batch_extract_pretrain.py --merge_only --input_gpkg x --output_folder x --chunk_idx 0 --total_chunks 0

# python batch_extract_pretrain.py \
#     --input_gpkg "$SCRATCH/ntems/sampling_plan_10k.gpkg" \
#     --output_folder "$SCRATCH/ntems/ontario_pretrain_npy" \
#     --total_chunks 200 \
#     --chunk_idx "$SLURM_ARRAY_TASK_ID" \
#     --num_workers 12
