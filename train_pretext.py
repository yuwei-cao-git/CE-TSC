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

from dataset.PretextDataset import PretextDataModule
from model.pretext import OntarioPretrainTask


def main():
    parser = argparse.ArgumentParser(description="Ontario Forest Pre-training")

    # Path Arguments
    parser.add_argument("--data_root", type=str, default="./data/ontario_pretrain_npy")
    parser.add_argument("--img_emb_dir", type=str)
    parser.add_argument("--train_csv", type=str, default="train_split.csv")
    parser.add_argument("--val_csv", type=str, default="val_split.csv")
    parser.add_argument("--test_csv", type=str, default="test_split.csv")
    parser.add_argument("--experiment_name", type=str)

    # Model Architecture (Critical for the PyPI PointNext)
    parser.add_argument(
        "--mode",
        type=str,
        default="pretext_both",
        choices=["pretext_both", "pretext_lsc", "pretext_both_emb", "pretext_lsc_emb"],
    )
    parser.add_argument("--encoder", type=str, default="b", choices=["s", "b", "l", "xl"])
    parser.add_argument("--pc_emb_dims", type=int, default=512, help="Latent dimension of backbone")
    parser.add_argument("--pc_emb_scale", type=int, default=2)
    parser.add_argument("--img_emb_dims", type=int, default=128)
    parser.add_argument("--num_species", type=int, default=16)
    parser.add_argument("--num_ecoregions", type=int, default=11)
    parser.add_argument("--eco_emb_dim", type=int, default=16, help="Ecoregion embedding size")
    parser.add_argument("--align_head", action="store_true")

    # Hyperparameters
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--lambda_struct", type=float, default=0.5)
    parser.add_argument("--dp_pc", type=float, default=0.3, help="Dropout rate")

    parser.add_argument("--rot", type=bool, default=False)
    parser.add_argument("--num_workers", type=int, default=6)

    args = parser.parse_args()

    # 2. Convert Namespace to Dictionary for Lightning
    config = vars(args)
    config["project_name"] = "Ontario_Forest_Pretrain"

    # 3. Initialize DataModule
    dm = PretextDataModule(config)

    # 4. Initialize Multi-Task Task
    model = OntarioPretrainTask(config)

    # 5. Setup W&B Logger (Logs the args automatically)
    wandb_logger = WandbLogger(
        project=config["project_name"],
        name=f"{args.experiment_name}_{args.encoder}{args.pc_emb_dims}_img{args.img_emb_dims}_{args.mode}_LR{args.lr}_L{args.lambda_struct}",
        save_dir=os.path.join(
            os.environ.get("SCRATCH", "."),
            "CE_logs",
            "pre_wandb",
        ),
        config=config,  # This ensures all hyperparams are tracked in the W&B dashboard
    )

    # 6. Checkpoint Callback (Saves to unique folder per experiment)
    early_stopping = EarlyStopping(
        monitor="val_loss",  # Metric to monitor
        patience=10,  # Number of epochs with no improvement after which training will be stopped
        mode="min",  # Set "min" for validation loss
        verbose=True,
    )
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(
            os.environ.get("SCRATCH", "."), "CE_logs", "pre_checkpoints", f"{args.experiment_name}"
        ),
        filename="best-{epoch:02d}-{val_acc:.2f}",
        monitor="val_acc",
        mode="max",
        save_top_k=1
    )

    # 7. Trainer
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        logger=wandb_logger,
        callbacks=[
            checkpoint_callback,
            early_stopping, LearningRateMonitor(logging_interval="step"),
        ],
        strategy="auto",
    )

    trainer.fit(model, datamodule=dm)

    trainer.test(model, datamodule=dm)


if __name__ == "__main__":
    main()
