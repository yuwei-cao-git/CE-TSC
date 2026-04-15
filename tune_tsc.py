import os
import argparse
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
    EarlyStopping,
)

from dataset.tsc_data import TSCDataModule
# from dataset.mapping_utils import get_mapping_matrix
from model.tsc_task import TSCTuningTask


def main():
    parser = argparse.ArgumentParser(description="Ontario Forest Stage B: TSC Fine-tuning")

    # Data Paths
    parser.add_argument("--data_dir", type=str, required=True, help="Path to site-specific npz files")
    parser.add_argument("--dataset", type=str, required=True, choices=["wrf_sp", "rmf_sp", "nif_sp", "ovf_sp"])

    # Checkpoint Path (From Stage A)
    parser.add_argument("--pretrained_ckpt", type=str, default=None, help="Path to .ckpt or leave empty for scratch")

    # Model Params (Must match Stage A)
    parser.add_argument("--encoder", type=str, default="b")
    parser.add_argument("--emb_dims", type=int, default=768)
    parser.add_argument("--num_species", type=int, default=16)
    parser.add_argument("--num_ecoregions", type=int, default=11)
    parser.add_argument("--eco_emb_dim", type=int, default=16, help="Ecoregion embedding size")
    parser.add_argument("--replace_head", action="store_true")
    parser.add_argument("--align_head", action="store_true")

    # Tuning Hyperparams
    parser.add_argument("--lr", type=float, default=1e-4, help="Lower LR for fine-tuning")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--num_workers", type=int, default=6)
    parser.add_argument("--loss_func", type=str, default="mse")
    parser.add_argument("--mode", type=str, default="downstream")

    args = parser.parse_args()
    config = vars(args)
    if config["dataset"] == "wrf_sp":
        class_weights = [
            0.00581433,
            0.04249615,
            0.01268365,
            0.01761785,
            0.02549208,
            0.04876563,
            0.09891814,
            0.74821219,
        ]
    elif config["dataset"] == "rmf_sp":
        class_weights = [
            0.13698052,
            0.02423971,
            0.05625495,
            0.04423566,
            0.0258036,
            0.03557034,
            0.25459799,
            0.00646059,
            0.41585663,
        ]
    elif config["dataset"] == "nif_sp":
        class_weights = [
            0.01336019,
            0.06994183,
            0.08378996,
            0.05661675,
            0.17902111,
            0.24095274,
            0.09358207,
            0.19145136,
            0.07128399,
        ]
    else:
        class_weights = [
            0.21605368,
            0.01386011,
            0.12620734,
            0.11190644,
            0.07048709,
            0.05551495,
            0.06225591,
            0.02889582,
            0.23440844,
            0.04299998,
            0.03741025,
        ]
    if config["loss_func"] in ["wmse", "ewmse"]:
        config["class_weights"] = torch.tensor(class_weights).float()
    else:
        config["class_weights"] = None
    if config["replace_head"]:
        config["num_species"] = len(class_weights)
    # A. Determine Site Name (e.g., 'WRF') and generate mapping matrix
    # site_key = args.dataset.split("_")[0].upper()
    # mapping_matrix = get_mapping_matrix(site_key)
    # print(f"--- Initializing for {site_key} with {config['num_species']} labels ---")

    pl.seed_everything(123)
    # 1. Setup Data
    # Note: We still pass the transform config if you want to use the same augmentations
    dm = TSCDataModule(config)

    # 2. Setup Task & Load Weights
    # The TSCTuningTask handles strict=False loading of the backbone
    model = TSCTuningTask(
        config=config,
        mapping_matrix=None,
        pretrained_path=args.pretrained_ckpt,
    )

    # 3. Logger & Callbacks
    wandb_logger = WandbLogger(
        project="Ontario_Forest_TSC_FineTune",
        name=f"FineTune_{args.dataset}_LR{args.lr}_ENC{args.encoder}{args.emb_dims}_ECO{args.eco_emb_dim}",
        save_dir=os.path.join(
            os.environ.get("SCRATCH", "."), "CE_logs", "tsc_wandb"
        ),
        config=config,
    )
    early_stopping = EarlyStopping(
        monitor="val_r2",  # Metric to monitor
        patience=10,  # Number of epochs with no improvement after which training will be stopped
        mode="max",  # Set "min" for validation loss
        verbose=True,
    )
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(
            os.environ.get("SCRATCH", "."), "CE_logs", "tsc_checkpoints", f"{args.dataset}"
        ),
        filename="best-tsc-{epoch:02d}-{val_r2:.3f}",
        monitor="val_r2",
        mode="max",
        save_top_k=1,
    )

    # 4. Trainer
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        gradient_clip_val=0.5,
        num_nodes=1,
        strategy="auto",
        logger=wandb_logger,
        callbacks=[
            checkpoint_callback,
            early_stopping,
            LearningRateMonitor(logging_interval="step"),
        ],
    )

    trainer.fit(model, datamodule=dm)


if __name__ == "__main__":
    main()
