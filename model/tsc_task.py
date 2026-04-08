import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics.regression import R2Score, MeanSquaredError
from .pointnext_ontario import PointNextOntario
from .model_utils import get_loss

class TSCTuningTask(pl.LightningModule):
    def __init__(self, config, mapping_matrix, pretrained_path=None):
        super().__init__()
        self.save_hyperparameters(ignore=["mapping_matrix"])
        self.config = config

        # Determine number of output classes for this site
        # num_site_labels will be 9 for NIF, 8 for WRF, etc.
        self.num_site_labels = mapping_matrix.shape[1] if config["replace_head"] else 16
        self.model_out_dim = self.num_site_labels if config["replace_head"] else 16

        # Initialize Model
        self.model = PointNextOntario(config, in_dim=3, num_species=self.model_out_dim, num_ecoregions=11)
        self.register_buffer("mapping_matrix", mapping_matrix)

        if pretrained_path:
            ckpt = torch.load(pretrained_path, map_location="cpu")
            self.model.load_state_dict(ckpt["state_dict"], strict=False)

            print(f"--- SUCCESS: Initialized from {pretrained_path} ---")
        else:
            print("--- NOTICE: No valid pre-trained path. Training FROM SCRATCH. ---")

        # Metrics: Use 'global' to avoid batch-size artifacts
        # self.val_r2 = R2Score(num_outputs=self.model_out_dim, multioutput="uniform_average")
        self.val_r2 = R2Score()
        self.val_rmse = MeanSquaredError(squared=False)

        # Loss: KL Divergence is best for proportions (0.0 to 1.0)
        # self.criterion = nn.KLDivLoss(reduction="batchmean")
        # self.criterion = nn.MSELoss()
        self.loss_func = config["loss_func"]
        self.weights = config["class_weights"]

    def forward(self, batch):
        pred = self.model(
            batch["pc_feat"].transpose(1, 2),
            batch["point_cloud"].transpose(1, 2),
            batch["ecoregion"] if self.config["eco_emb_dim"] != 0 else -1,
            mode="downstream",
        )
        # --- LOGIC SWITCH ---
        if self.config["replace_head"]:
            # Model output already matches site labels
            pred_site = pred
        else:
            # Model output is 16, must collapse via matrix
            pred_site = torch.matmul(pred, self.mapping_matrix)
            # Ensure normalization after matrix multiplication
            pred_site = pred_site / (pred_site.sum(dim=-1, keepdim=True) + 1e-10)

        return pred_site

    def training_step(self, batch, batch_idx):
        pred_site = self.forward(batch)
        # Use log_softmax for numerical stability
        # loss = self.criterion(pred_site, batch["label"])
        loss = get_loss(self.loss_func, pred_site, batch["label"], self.weights)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        pred_site = self.forward(batch)
        target = batch["label"]

        self.val_r2.update(torch.round(pred_site, decimals=1).view(-1), target.view(-1))
        self.val_rmse.update(pred_site, target)

        # loss = self.criterion(pred_site, target)
        loss = get_loss(self.loss_func, pred_site, batch["label"], self.weights)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, sync_dist=True)

    def on_validation_epoch_end(self):
        r2 = self.val_r2.compute()
        rmse = self.val_rmse.compute()

        # Log metrics. Clamp R2 for display, but keep raw value in WandB
        self.log("val_r2", r2, prog_bar=True, sync_dist=True)
        self.log("val_rmse", rmse, prog_bar=True, sync_dist=True)

        self.val_r2.reset()
        self.val_rmse.reset()

    def configure_optimizers(self):
        # Use AdamW with a slightly higher weight decay for fine-tuning
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.config.get("lr", 1e-4), weight_decay=0.0001
        )

        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-6)
        # return {"optimizer": optimizer,"lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"}}

        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        return [optimizer], [scheduler]
