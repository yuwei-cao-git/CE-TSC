import os, argparse, numpy as np, pandas as pd, geopandas as gpd, pdal, json, subprocess, re, sys
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from pyproj import Transformer

def log_message(message, log_file):
    with open(log_file, "a") as f:
        f.write(message + "\n")
    print(message)
    sys.stdout.flush()

def get_tile_info(laz_path):
    try:
        p = pdal.Pipeline(json.dumps([{"type": "readers.las", "filename": str(laz_path), "count": 1}]))
        p.execute()
        meta = p.metadata
        if isinstance(meta, str): meta = json.loads(meta)
        b = meta['metadata']['readers.las']
        bounds = {"minx": b['minx'], "maxx": b['maxx'], "miny": b['miny'], "maxy": b['maxy']}
        srs = meta['metadata']['readers.las']['srs']['compoundwkt']
        # Extract EPSG code
        match = re.search(r'EPSG",(\d+)', srs)
        epsg = match.group(1) if match else "2959"
        return epsg, bounds
    except: return "2959", None

def process_single_plot(laz_path, row, output_folder, transformer, target_n=7168, debug_log=None):
    try:
        # TRANSFORM using the manual math bridge
        cx_tile, cy_tile = transformer.transform(row["x"], row["y"])
        species_id = int(row["label"])

        if debug_log:
            log_message(f"    [COORD] Lambert({row['x']:.1f}) -> UTM({cx_tile:.1f})", debug_log)

        p_json = [
            {"type": "readers.las", "filename": str(laz_path)},
            {"type": "filters.crop", "point": f"POINT({cx_tile:.3f} {cy_tile:.3f})", "distance": 11.28},
            {"type": "filters.range", "limits": "Z(2:)"}
        ]
        pipe = pdal.Pipeline(json.dumps(p_json))
        pipe.execute()
        
        if not pipe.arrays: return "EMPTY"
        pts = pipe.arrays[0]
        if len(pts) < target_n: return "DENSITY"

        data = np.vstack((pts["X"] - cx_tile, pts["Y"] - cy_tile, pts["Z"])).T
        idx = np.random.choice(data.shape[0], target_n, replace=False)
        
        npy_path = Path(output_folder) / str(species_id) / f"{species_id}_{int(row['x'])}_{int(row['y'])}.npy"
        npy_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(npy_path), data[idx].astype(np.float32))
        return "SUCCESS"
    except Exception as e: return f"ERROR_{e}"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_gpkg", type=True, required=True)
    parser.add_argument("--output_folder", type=True, required=True)
    parser.add_argument("--total_chunks", type=int, required=True)
    parser.add_argument("--chunk_idx", type=int, required=True)
    parser.add_argument("--num_workers", type=int, default=8)
    args = parser.parse_args()

    chunk_log = f"log_chunk_{args.chunk_idx}.txt"
    log_message(f"--- Job Chunk {args.chunk_idx} Started ---", chunk_log)
    
    out_dir = Path(args.output_folder).resolve()
    gdf = gpd.read_file(args.input_gpkg, layer="sampling_plan_10k")
    chunk_tiles = np.array_split(gdf["Tilename"].unique(), args.total_chunks)[args.chunk_idx]
    
    tmp_dir = Path(os.environ.get("SLURM_TMPDIR", "/tmp")) / f"batch_{args.chunk_idx}"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # 3978 Proj String (Mathematical Definition)
    p_3978 = "+proj=lcc +lat_0=49 +lon_0=-95 +lat_1=49 +lat_2=77 +x_0=0 +y_0=0 +ellps=GRS80 +units=m +no_defs"

    # EPSG -> UTM Zone Mapping
    utm_map = {"3159":"15", "3160":"16", "2958":"17", "2959":"18"}

    stats = {"SUCCESS": 0, "EMPTY": 0, "DENSITY": 0}

    for i, tile_name in enumerate(chunk_tiles):
        tile_df = gdf[gdf["Tilename"] == tile_name]
        url = tile_df["Download_H"].iloc[0]
        local_laz = tmp_dir / url.split('/')[-1]

        log_message(f"Tile {i}: {tile_name}", chunk_log)
        subprocess.run(["wget", "-q", "-O", str(local_laz), url])
        if not local_laz.exists(): continue

        # Detect Zone and Setup Manual Transformer
        epsg_code, bounds = get_tile_info(local_laz)
        zone = utm_map.get(epsg_code, "17")
        p_utm = f"+proj=utm +zone={zone} +ellps=GRS80 +units=m +no_defs"
        
        log_message(f"  [INFO] Native EPSG: {epsg_code} (Zone {zone}N) | Bounds: X[{bounds['minx']:.0f}]", chunk_log)
        transformer = Transformer.from_crs(p_3978, p_utm, always_xy=True)

        with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
            futures = [executor.submit(process_single_plot, local_laz, r, out_dir, transformer, 7168, (chunk_log if j==0 else None)) 
                       for j, (_, r) in enumerate(tile_df.iterrows())]
            for f in futures:
                res = f.result()
                if res == "SUCCESS": stats["SUCCESS"] += 1
                elif res == "EMPTY": stats["EMPTY"] += 1
                elif res == "DENSITY": stats["DENSITY"] += 1

        local_laz.unlink()
        log_message(f"  Current Progress: Saved={stats['SUCCESS']} | Empty={stats['EMPTY']}", chunk_log)

if __name__ == "__main__":
    main()