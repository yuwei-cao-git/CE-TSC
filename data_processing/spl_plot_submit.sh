#!/bin/bash
#SBATCH --job-name=ont_pretrain
#SBATCH --output=logs/ont_%A_%a.out
#SBATCH --error=logs/ont_%A_%a.err
#SBATCH --array=0-199
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=10:00:00

set -euo pipefail

# 1. Setup absolute paths
OUTPUT_DIR="$SCRATCH/ntems/ontario_pretrain_npy"
mkdir -p logs "$OUTPUT_DIR"

# 2. Environment Setup (Compute Canada/Alliance style)
module load python/3.10 pdal scipy-stack

rm -rf $SLURM_TMPDIR/env

# 3. Virtual Environment in Job-Specific Local Storage
# This prevents 100 jobs from clashing while reading the same env files
virtualenv --no-download "$SLURM_TMPDIR/env"
source "$SLURM_TMPDIR/env/bin/activate"

# 4. Install requirements (Use --no-index to use cluster's pre-cached wheels)
pip install --no-index --upgrade pip
pip install numpy pyproj 
pip install --no-index pandas geopandas tqdm 
# If PDAL isn't in the scipy-stack, install it via pip:
pip install --no-index pdal || pip install pdal

# 5. Run with Unbuffered Output
# PYTHONUNBUFFERED=1 ensures prints show up in .out files immediately
# We pass --num_workers 8 to avoid I/O bottlenecks on the node SSD
export PYTHONUNBUFFERED=1

srun python batch_extract_pretrain.py \
    --input_gpkg "$SCRATCH/ntems/sampling_plan_10k.gpkg" \
    --output_folder "$OUTPUT_DIR" \
    --total_chunks 200\
    --chunk_idx "$SLURM_ARRAY_TASK_ID" \
    --num_workers 12

# Merge all batch files into one master manifest
awk 'FNR==1 && NR!=1{next;}{print}' ./logs/meta_batch_*.csv > training_master_list.csv

# Verify the count (should be ~126,581)
wc -l training_master_list.csv
