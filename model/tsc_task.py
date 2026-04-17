import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics.regression import R2Score, MeanSquaredError
from .pointnext_ontario import PointNextOntario
from .model_utils import get_loss, initialize_weights


class TSCTuningTask(pl.LightningModule):
    def __init__(self, config, mapping_matrix=None, pretrained_path=None):
        super().__init__()
        self.save_hyperparameters(ignore=["mapping_matrix"])
        self.config = config

        # Determine number of output classes for this site
        # num_site_labels will be 9 for NIF, 8 for WRF, etc.
        if self.config["replace_head"]:
            self.model_out_dim = config["num_species"] if config["replace_head"] else 16
        else:
            self.model_out_dim = mapping_matrix.shape[1] if config["replace_head"] else 16
            self.register_buffer("mapping_matrix", mapping_matrix)

        print(f"num_species: {config['num_species']}")

        # Initialize Model
        self.model = PointNextOntario(config, in_dim=3, num_species=self.model_out_dim, num_ecoregions=11)

        if pretrained_path:
            ckpt = torch.load(pretrained_path, map_location="cpu")
            self.model.load_state_dict(ckpt["state_dict"], strict=False)

            print(f"--- SUCCESS: Initialized from {pretrained_path} ---")
        else:
            initialize_weights(self.model)
            print("--- NOTICE: No valid pre-trained path. Training FROM SCRATCH. ---")

        # Metrics
        self.val_r2 = R2Score()
        self.val_rmse = MeanSquaredError(squared=False)

        # Loss
        # self.criterion = nn.KLDivLoss(reduction="batchmean")
        # self.criterion = nn.MSELoss()
        self.loss_func = config["loss_func"]
        self.weights = config["class_weights"]

    def forward(self, batch):
        if "both" in self.config.get("mode", ""):
            dom_logits, comp_pred = self.model(
                batch["point_cloud"].transpose(1, 2),
                batch["pc_feat"].transpose(1, 2),
                batch["ecoregion"] if self.config["eco_emb_dim"] > 0 else None,
                (
                    batch["patch_embed"]
                    if "embedding" in self.config.get("mode", "")
                    else None
                ),
                mode=self.config["mode"],
            )
        else:
            comp_pred = self.model(
                batch["point_cloud"].transpose(1, 2),
                batch["pc_feat"].transpose(1, 2),
                batch["ecoregion"] if self.config["eco_emb_dim"] > 0 else None,
                (
                    batch["patch_embed"]
                    if "embedding" in self.config.get("mode", "")
                    else None
                ),
                mode=self.config["mode"],
            )

        self.weights = (
            self.weights.to(comp_pred.device) if self.weights is not None else None
        )
        # --- LOGIC SWITCH ---
        if self.config["replace_head"]:
            # Model output already matches site labels
            pred_site = comp_pred
        else:
            # Model output is 16, must collapse via matrix
            pred_site = torch.matmul(comp_pred, self.mapping_matrix)
            # Ensure normalization after matrix multiplication
            pred_site = pred_site / (pred_site.sum(dim=-1, keepdim=True) + 1e-10)

        if "both" in self.config.get("mode", ""):
            return dom_logits, pred_site
        else:
            return pred_site

    def training_step(self, batch, batch_idx):
        if "both" in self.config.get("mode", ""):
            dom_logits, pred_site = self.forward(batch)
        else:
            pred_site = self.forward(batch)
        # Use log_softmax for numerical stability
        # loss = self.criterion(pred_site, batch["label"])
        loss_comp = get_loss(self.loss_func, pred_site, batch["label"], self.weights)
        loss_cls = F.cross_entropy(dom_logits, torch.argmax(batch["label"], dim=1))  if self.config["mode"] == "downstream_both" else 0.0
        loss = 0.01 * loss_cls + loss_comp
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        if "both" in self.config.get("mode", ""):
            dom_logits, pred_site = self.forward(batch)
        else:
            pred_site = self.forward(batch)
        target = batch["label"]
        # Use log_softmax for numerical stability
        # loss = self.criterion(pred_site, batch["label"])
        loss_comp = get_loss(self.loss_func, pred_site, target, self.weights)
        loss_cls = (
            F.cross_entropy(dom_logits, torch.argmax(target, dim=1))
            if self.config["mode"] == "downstream_both"
            else 0.0
        )
        loss = 0.01 * loss_cls + loss_comp

        self.val_r2.update(torch.round(pred_site, decimals=2).view(-1), target.view(-1))
        self.val_rmse.update(pred_site, target)

        # loss = self.criterion(pred_site, target)
        self.log(
            "val_comp_loss", loss_comp, on_epoch=True, prog_bar=True, sync_dist=True
        )
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
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
