import os, argparse, numpy as np, pandas as pd, geopandas as gpd, pdal, json, subprocess, re, sys
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from pyproj import Transformer

# Custom logger to force immediate writing to a text file
def log_message(message, log_file):
    with open(log_file, "a") as f:
        f.write(message + "\n")
    print(message) # Still print to .out
    sys.stdout.flush()

def get_native_epsg(laz_path):
    try:
        pipeline = pdal.Pipeline(json.dumps([{"type": "readers.las", "filename": str(laz_path), "count": 1}]))
        pipeline.execute()
        meta = pipeline.metadata
        if isinstance(meta, str): meta = json.loads(meta)
        srs = meta['metadata']['readers.las']['srs']['compoundwkt']
        match = re.search(r'EPSG",(\d+)', srs)
        return f"EPSG:{match.group(1)}" if match else "EPSG:2959"
    except: return "EPSG:2959"

def process_single_plot(laz_path, row, output_folder, transformer, target_n=7168):
    try:
        cx_tile, cy_tile = transformer.transform(row["x"], row["y"])
        species_id = int(row["label"])

        p_json = [
            {"type": "readers.las", "filename": str(laz_path)},
            {"type": "filters.crop", "point": f"POINT({cx_tile} {cy_tile})", "distance": 11.28},
            {"type": "filters.range", "limits": "Z(2:)"}
        ]
        pipe = pdal.Pipeline(json.dumps(p_json))
        pipe.execute()
        
        if not pipe.arrays: return "SKIP_EMPTY"
        pts = pipe.arrays[0]
        if len(pts) < target_n: return f"SKIP_DENSITY_{len(pts)}"

        # Save logic
        data = np.vstack((pts["X"] - cx_tile, pts["Y"] - cy_tile, pts["Z"])).T
        idx = np.random.choice(data.shape[0], target_n, replace=False)
        
        npy_path = Path(output_folder) / str(species_id) / f"{species_id}_{int(row['x'])}_{int(row['y'])}.npy"
        npy_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(npy_path), data[idx].astype(np.float32))
        return "SUCCESS"
    except Exception as e:
        return f"ERROR_{str(e)}"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_gpkg", type=str, required=True)
    parser.add_argument("--output_folder", type=str, required=True)
    parser.add_argument("--total_chunks", type=int, required=True)
    parser.add_argument("--chunk_idx", type=int, required=True)
    args = parser.parse_args()

    # Define a log file specific to this chunk
    chunk_log = f"log_chunk_{args.chunk_idx}.txt"
    log_message(f"--- Starting Job Chunk {args.chunk_idx} ---", chunk_log)

    out_dir = Path(args.output_folder).resolve()
    gdf = gpd.read_file(args.input_gpkg, layer="sampling_plan_10k")
    unique_tiles = gdf["Tilename"].unique()
    chunk_tiles = np.array_split(unique_tiles, args.total_chunks)[args.chunk_idx]
    
    tmp_dir = Path(os.environ.get("SLURM_TMPDIR", "/tmp")) / f"batch_{args.chunk_idx}"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    
    stats = {"SUCCESS": 0, "EMPTY": 0, "LOW_DENSITY": 0, "ERR": 0}

    for i, tile_name in enumerate(chunk_tiles):
        tile_df = gdf[gdf["Tilename"] == tile_name]
        url = tile_df["Download_H"].iloc[0]
        local_laz = tmp_dir / url.split('/')[-1]

        # Download
        log_message(f"Tile {i}: Downloading {tile_name}...", chunk_log)
        dl = subprocess.run(["wget", "-q", "-O", str(local_laz), url])
        
        if not local_laz.exists() or local_laz.stat().st_size < 1000:
            log_message(f"  [!] Download Failed for {url}", chunk_log)
            continue

        native_epsg = get_native_epsg(local_laz)
        trans = Transformer.from_crs("EPSG:3978", native_epsg, always_xy=True)

        # Process
        with ThreadPoolExecutor(max_workers=8) as executor: # Reduced workers for stability
            futures = [executor.submit(process_single_plot, local_laz, r, out_dir, trans) for _, r in tile_df.iterrows()]
            for f in futures:
                res = f.result()
                if res == "SUCCESS": stats["SUCCESS"] += 1
                elif "SKIP_EMPTY" in res: stats["EMPTY"] += 1
                elif "SKIP_DENSITY" in res: stats["LOW_DENSITY"] += 1
                else: 
                    stats["ERR"] += 1
                    log_message(f"  [!] Worker Error: {res}", chunk_log)

        if local_laz.exists(): local_laz.unlink()

        # Immediate Status Update
        log_message(f"  Summary: Saved={stats['SUCCESS']} | Empty={stats['EMPTY']} | LowDensity={stats['LOW_DENSITY']} | Errors={stats['ERR']}", chunk_log)

if __name__ == "__main__":
    main()