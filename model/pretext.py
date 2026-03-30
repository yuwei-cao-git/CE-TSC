import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from pointnext_ontario import PointNeXtOntario # The model we just built

class OntarioPretrainTask(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        
        # Initialize model
        self.model = PointNeXtOntario(
            num_species=config['num_species'],
            num_ecoregions=config['num_ecoregions'],
            emb_dim=config.get('emb_dim', 16)
        )
        
        # Loss Functions
        self.ce_loss = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()
        
        # Weight for the structural regression task
        self.lambda_struct = config.get('lambda_struct', 0.5)

    def forward(self, batch):
        return self.model(batch, mode='pretext')

    def training_step(self, batch, batch_idx):
        outputs = self(batch)
        
        # 1. Species Classification Loss (Weak Supervision)
        loss_species = self.ce_loss(outputs['species_logits'], batch['species_label'])
        
        # 2. Structural Regression Loss (H95 Proxy)
        # We normalize the target if needed, but MSE on raw meters is fine if Z is scaled
        loss_struct = self.mse_loss(outputs['h95_pred'], batch['structure_label'])
        
        # Total Weighted Loss
        total_loss = loss_species + (self.lambda_struct * loss_struct)
        
        # Logging for Monitoring
        self.log('train_loss', total_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_species_loss', loss_species, on_epoch=True)
        self.log('train_h95_rmse', torch.sqrt(loss_struct), on_epoch=True)
        
        return total_loss

    def validation_step(self, batch, batch_idx):
        outputs = self(batch)
        
        loss_species = self.ce_loss(outputs['species_logits'], batch['species_label'])
        loss_struct = self.mse_loss(outputs['h95_pred'], batch['structure_label'])
        
        val_loss = loss_species + (self.lambda_struct * loss_struct)
        
        # Calculate Accuracy for Species
        preds = torch.argmax(outputs['species_logits'], dim=1)
        acc = (preds == batch['species_label']).float().mean()
        
        self.log('val_loss', val_loss, on_epoch=True, prog_bar=True)
        self.log('val_acc', acc, on_epoch=True, prog_bar=True)
        self.log('val_h95_rmse', torch.sqrt(loss_struct), on_epoch=True)
        
        return val_loss

    def configure_optimizers(self):
        # PointNeXt usually benefits from AdamW with Weight Decay
        optimizer = optim.AdamW(
            self.parameters(), 
            lr=self.config['lr'], 
            weight_decay=self.config.get('weight_decay', 1e-4)
        )
        
        # Cosine Annealing is standard for high-performance point cloud models
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.config['max_epochs']
        )
        
        return [optimizer], [scheduler]