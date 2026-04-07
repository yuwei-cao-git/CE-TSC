import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics.regression import R2Score, MeanSquaredError


class TSCTuningTask(pl.LightningModule):
    def __init__(self, config, mapping_matrix, pretrained_path=None):
        super().__init__()
        self.save_hyperparameters(ignore=["mapping_matrix"])
        self.config = config

        # Model
        from .pointnext_ontario import PointNextOntario

        self.model = PointNextOntario(
            config, in_dim=3, num_species=16, num_ecoregions=11
        )
        self.register_buffer("mapping_matrix", mapping_matrix)

        if pretrained_path:
            ckpt = torch.load(pretrained_path, map_location="cpu")
            self.model.load_state_dict(ckpt["state_dict"], strict=False)
            print(f"--- SUCCESS: Initialized from {pretrained_path} ---")
        else:
            print("--- NOTICE: No valid pre-trained path. Training FROM SCRATCH. ---")

        # Metrics: Use 'global' to avoid batch-size artifacts
        num_site_labels = mapping_matrix.shape[1]
        self.val_r2 = R2Score(
            num_outputs=num_site_labels, multioutput="uniform_average"
        )
        self.val_rmse = MeanSquaredError(squared=False)

        # Loss: KL Divergence is best for proportions (0.0 to 1.0)
        # self.criterion = nn.KLDivLoss(reduction="batchmean")
        self.criterion = nn.MSELoss()

    def forward(self, batch):
        pred_16 = self.model(
            batch["pc_feat"].transpose(1, 2),
            batch["point_cloud"].transpose(1, 2),
            batch["ecoregion"],
            mode="downstream",
        )
        # Collapse 16 -> Site Labels
        pred_site = torch.matmul(pred_16, self.mapping_matrix)
        # Final Softmax safety (ensures sum to 1.0 after mapping)
        pred_site = pred_site / (pred_site.sum(dim=-1, keepdim=True) + 1e-10)
        return pred_site

    def training_step(self, batch, batch_idx):
        pred_site = self.forward(batch)
        # Use log_softmax for numerical stability
        loss = self.criterion(torch.log(pred_site + 1e-10), batch["label"])
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        pred_site = self.forward(batch)
        target = batch["label"]

        self.val_r2.update(pred_site, target)
        self.val_rmse.update(pred_site, target)

        loss = self.criterion(torch.log(pred_site + 1e-10), target)
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
            self.parameters(), lr=self.config.get("lr", 1e-4), weight_decay=0.05
        )
        # Cosine Annealing helps the R2 flip from negative to positive smoothly
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=10, gamma=0.1
        )
        return [optimizer], [scheduler]
