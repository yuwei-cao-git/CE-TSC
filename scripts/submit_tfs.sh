#!/bin/bash
#SBATCH --nodes=1               # number of nodes
#SBATCH --job-name=ont_tune
#SBATCH --array=0-3            # Launch 4 experiments
#SBATCH --gpus=nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=24:00:00

# Trap the exit status of the job
trap 'job_failed=$?' EXIT
# code transfer
cd $SLURM_TMPDIR
mkdir -p work/CE-TSC
cd work/CE-TSC
git clone https://github.com/yuwei-cao-git/CE-TSC.git
echo "Source code cloned!"

# data transfer
mkdir -p data
# extract an archive to a different directory, the ‘-C’ option is followed by the destination path
tar -I pigz -xf $project/TL-TSC/data/wrf_superpixel_dataset.tar.gz -C ./data || { echo "wrf extract failed"; exit 1; }
echo "Data transfered"

# Load python module, and additional required modules
module load StdEnv/2023 cuda/12.6 python/3.11 gcc/12.3 arrow/21.0
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip
pip install --no-index torch==2.5.0 pointnext==0.0.5
pip install --no-index tensorboardX lightning==2.5.3 pytorch_lightning==2.5.3 torchaudio==2.5.0 torchdata torcheval torchmetrics torchtext torchvision==0.20.0 rasterio imageio wandb pandas
pip install --no-index scikit-learn seaborn open3d==0.18.0
echo "Virtual Env created!"

# Set environment variables
export TORCH_NCCL_ASYNC_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=1  #Set this environment variable if you wish to use the NCCL backend for inter-GPU communication.
export MASTER_ADDR=$(hostname) #Store the master node’s IP address in the MASTER_ADDR environment variable.

# Log experiment variables
export WANDB_API_KEY=
wandb offline

# Define your hyperparameter grid
EMB=(512 768 768 1024)
ENC=(s b l xl)

# Select the parameters for THIS array task
CUR_EMB=${LRS[$SLURM_ARRAY_TASK_ID]}
CUR_ENC=${ENC[$SLURM_ARRAY_TASK_ID]}

echo "Running Experiment $SLURM_ARRAY_TASK_ID: LR=$CUR_LR, Lambda=$CUR_LAMBDA"

python train_pretext.py \
    --experiment_name "Tune_wrf_tfs_Array_$SLURM_ARRAY_TASK_ID" \
    --dataset "wrf_sp" \
    --data_dir "./data/wrf_superpixel_dataset" \
    --lr 0.0001 \
    --emb_dims $CUR_EMB \
    --encoder $CUR_ENC \
    --batch_size 128 \
    --max_epochs 200 \
    --replace_head

cp -r ./checkpoints ~/scratch/CE_logs/

echo "theend"