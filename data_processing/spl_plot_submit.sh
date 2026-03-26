#!/bin/bash
#SBATCH --job-name=ont_pretrain_extract
#SBATCH --output=logs/ont_%A_%a.out
#SBATCH --error=logs/ont_%A_%a.err
#SBATCH --array=0-1                # Adjust based on tile count (100 parallel chunks)
#SBATCH --cpus-per-task=16          # Multi-threading for PDAL/Python
#SBATCH --mem=64G                   # Memory for tile processing
#SBATCH --time=0:30:00             # Downloading/processing takes time

set -euo pipefail

# Create log and output directories
mkdir -p logs
OUTPUT_DIR="$SCRATCH/ntems/ontario_pretrain_npy"
mkdir -p "$OUTPUT_DIR"

# Setup Environment
module load python/3.10 pdal 

virtualenv --no-download "$SLURM_TMPDIR/env"
source "$SLURM_TMPDIR/env/bin/activate"

# Install requirements to the local node environment
pip install --no-index --upgrade pip
pip install --no-index numpy pandas geopandas tqdm 
pip install pdal

# Run the batch script
# We pass the index and total chunks to split the unique tiles
srun python batch_extract_pretrain.py \
    --input_gpkg "$SCRATCH/ntems/sampling_plan_10k.gpkg" \
    --output_folder "$OUTPUT_DIR" \
    --total_chunks 2 \
    --chunk_idx "$SLURM_ARRAY_TASK_ID" \
    --num_workers 16

# Count how many .npy files have been created so far
find $SCRATCH/ntems/ontario_pretrain_npy -name "*.npy" | wc -l

# Check the tail of a log file to see the progress bars
tail -f logs/ont_XXXX_XX.out
<<<<<<< HEAD
=======

# Merge all batch files into one master manifest
awk 'FNR==1 && NR!=1{next;}{print}' meta_batch_*.csv > training_master_list.csv

# Verify the count (should be ~126,581)
wc -l training_master_list.csv
>>>>>>> c0a7cc8dd315697f6a494d69609fd75118904c79
