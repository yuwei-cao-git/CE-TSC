import os
import argparse
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

from dataset.PretextDataset import PretextDataModule
from model.pretext import OntarioPretrainTask

def main():
    # 1. Setup Argument Parser
    parser = argparse.ArgumentParser(description="Ontario Forest Pre-training")
    
    # Path Arguments
    parser.add_argument("--data_root", type=str, default=os.path.join(os.environ.get('SCRATCH', '.'), "ntems/ontario_pretrain_npy"))
    parser.add_argument("--train_csv", type=str, default="train_split.csv")
    parser.add_argument("--val_csv", type=str, default="val_split.csv")
    
    # Model Architecture
    parser.add_argument("--num_species", type=int, default=12)
    parser.add_argument("--num_ecoregions", type=int, default=14)
    parser.add_argument("--emb_dim", type=int, default=16)
    
    # Hyperparameters (The "Tuning" variables)
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--lambda_struct", type=float, default=0.5, help="Weight for H95 loss (0.0 to 1.0)")
    
    # Cluster/Hardware
    parser.add_argument("--num_workers", type=int, default=12)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--experiment_name", type=str, default="PointNeXt_Pretrain")

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
        name=f"{args.experiment_name}_LR{args.lr}_L{args.lambda_struct}",
        config=config # This ensures all hyperparams are tracked in the W&B dashboard
    )

    # 6. Checkpoint Callback (Saves to unique folder per experiment)
    ckpt_path = os.path.join(os.environ.get('SCRATCH', '.'), "checkpoints", args.experiment_name)
    checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_path,
        filename="best-{epoch:02d}-{val_acc:.2f}",
        monitor="val_acc",
        mode="max",
        save_top_k=2
    )

    # 7. Trainer
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="gpu",
        devices=args.gpus,
        logger=wandb_logger,
        callbacks=[checkpoint_callback, LearningRateMonitor(logging_interval='step')],
        precision=16,
        strategy="ddp" if args.gpus > 1 else "auto"
    )

    trainer.fit(model, datamodule=dm)

if __name__ == "__main__":
    main()