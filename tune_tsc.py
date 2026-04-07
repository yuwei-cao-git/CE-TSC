import os
import argparse
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

from dataset.tsc_data import TSCDataModule
from dataset.mapping_utils import get_mapping_matrix
from model.tsc_task import TSCTuningTask


def main():
    parser = argparse.ArgumentParser(
        description="Ontario Forest Stage B: TSC Fine-tuning"
    )

    # Data Paths
    parser.add_argument(
        "--data_dir", type=str, required=True, help="Path to site-specific npz files"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["wrf_sp", "rmf_sp", "nif_sp", "ovf_sp"],
    )

    # Checkpoint Path (From Stage A)
    parser.add_argument(
        "--pretrained_ckpt",
        type=str,
        default=None,
        help="Path to .ckpt or leave empty for scratch",
    )

    # Model Params (Must match Stage A)
    parser.add_argument("--encoder", type=str, default="s")
    parser.add_argument("--emb_dims", type=int, default=512)
    parser.add_argument("--num_species", type=int, default=16)
    parser.add_argument("--num_ecoregions", type=int, default=11)

    # Tuning Hyperparams
    parser.add_argument(
        "--lr", type=float, default=1e-4, help="Lower LR for fine-tuning"
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--num_workers", type=int, default=6)

    args = parser.parse_args()
    config = vars(args)

    # A. Determine Site Name (e.g., 'WRF') and generate mapping matrix
    site_key = args.dataset.split("_")[0].upper()
    mapping_matrix = get_mapping_matrix(site_key)
    print(f"--- Initializing for {site_key} with {mapping_matrix.shape[1]} labels ---")
    # 1. Setup Data
    # Note: We still pass the transform config if you want to use the same augmentations
    dm = TSCDataModule(config)

    # 2. Setup Task & Load Weights
    # The TSCTuningTask handles strict=False loading of the backbone
    model = TSCTuningTask(
        config=config,
        mapping_matrix=mapping_matrix,
        pretrained_path=args.pretrained_ckpt,
    )

    # 3. Logger & Callbacks
    wandb_logger = WandbLogger(
        project="Ontario_Forest_TSC_FineTune",
        name=f"FineTune_{args.dataset}_LR{args.lr}",
        config=config,
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(
            os.environ.get("SCRATCH", "."), "checkpoints", f"tsc_{args.dataset}"
        ),
        filename="best-tsc-{epoch:02d}-{val_rmse:.3f}",
        monitor="val_rmse",
        mode="min",
        save_top_k=1,
    )

    # 4. Trainer
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="gpu",
        devices=1,  # Site-specific datasets are smaller; 1 GPU is usually enough
        logger=wandb_logger,
        callbacks=[checkpoint_callback, LearningRateMonitor(logging_interval="step")]
    )

    trainer.fit(model, datamodule=dm)


if __name__ == "__main__":
    main()
