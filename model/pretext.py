import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from .pointnext_ontario import PointNextOntario
from .model_utils import get_loss, initialize_weights


class OntarioPretrainTask(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.config = config

        # Initialize model using the pointnext pypi-based class
        # in_dim=3 (for standardized x, y, z features)
        self.model = PointNextOntario(
            config=config,
            in_dim=3,
            num_species=config["num_species"],
            num_ecoregions=config["num_ecoregions"]
        )
        initialize_weights(self.model)

        # Loss Functions
        self.ce_loss = nn.CrossEntropyLoss()

        if "both" in self.config.get("mode", ""):
            self.mse_loss = nn.MSELoss()
            # Weight for the structural regression task
            self.lambda_struct = config.get("lambda_struct", 0.5)

    def forward(self, batch):
        # PointNext package expects (B, C, N)
        # Our Dataset returns (B, N, C)
        pc_feat = batch["pc_feat"].transpose(1, 2)  # Standardized coords
        xyz = batch["point_cloud"].transpose(1, 2)  # Centered coords for grouping
        eco_idx = batch["ecoregion"] if self.config["eco_emb_dim"] > 0 else None
        patch_emb = batch["patch_embed"] if "emb" in self.config.get("mode", "") else None

        return self.model(xyz, pc_feat, eco_idx, patch_emb, mode=self.config["mode"])

    def training_step(self, batch, batch_idx):
        total_loss = 0.0
        if "both" in self.config.get("mode", ""):
            species_logits, h95_pred = self.forward(batch)
            # 2. Structural Regression Loss
            loss_struct = self.mse_loss(h95_pred, torch.log1p(batch["structure_label"]))
            # Total Weighted Loss
            total_loss = self.lambda_struct * loss_struct
        else:
            species_logits = self.forward(batch)

        # 1. Species Classification Loss
        loss_species = self.ce_loss(species_logits, batch["species_label"])
        total_loss += loss_species

        # Logging
        self.log(
            "train_loss",
            total_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        self.log("train_species_loss", loss_species, on_epoch=True, sync_dist=True)
        if "both" in self.config.get("mode", ""):
            self.log("train_h95_rmse", torch.sqrt(loss_struct), on_epoch=True, sync_dist=True)

        return total_loss

    def validation_step(self, batch, batch_idx):
        val_loss = 0.0
        if "both" in self.config.get("mode", ""):
            species_logits, h95_pred = self.forward(batch)
            loss_struct = self.mse_loss(h95_pred, torch.log1p(batch["structure_label"]))
            val_loss = self.lambda_struct * loss_struct
        else: 
            species_logits = self.forward(batch)

        loss_species = self.ce_loss(species_logits, batch["species_label"])
        val_loss += loss_species

        # Calculate Accuracy for Species
        preds = torch.argmax(species_logits, dim=1)
        acc = (preds == batch["species_label"]).float().mean()

        self.log("val_loss", val_loss, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val_acc", acc, on_epoch=True, prog_bar=True, sync_dist=True)
        if self.config["mode"] == "pretext_both":
            self.log("val_h95_rmse", torch.sqrt(loss_struct), on_epoch=True, sync_dist=True)

        return val_loss

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.parameters(),
            lr=self.config["lr"],
            weight_decay=self.config.get("weight_decay", 1e-4),
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=30, gamma=0.1
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
