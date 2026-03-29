import torch
import torch.nn as nn
import torch.nn.functional as F
from openpoints.models import build_model_from_cfg
from openpoints.utils import EasyConfig

class PointNeXtOntario(nn.Module):
    def __init__(self, num_species, num_ecoregions, emb_dim=16):
        super().__init__()
        
        # 1. PointNeXt Backbone (Encoder)
        # We use the standard PointNeXt-S or M configuration
        cfg = EasyConfig()
        cfg.model = {
            'NAME': 'BaseModel',
            'encoder_args': {
                'NAME': 'PointNextEncoder',
                'blocks': [1, 1, 1, 1, 1], # S-version, can be [1, 1, 1, 1, 1] for M
                'strides': [1, 2, 2, 2, 2],
                'width': 32,
                'in_channels': 3,
                'radius': 2.5, # Radius in meters
                'radius_scaling': 2.5,
                'sa_layers': 2,
                'sa_use_res': True,
            },
            'cls_args': {
                'NAME': 'ClsHead',
                'num_classes': 512, # Global latent feature size
                'mlps': [512, 256],
            }
        }
        self.backbone = build_model_from_cfg(cfg.model)
        
        # 2. Ecoregion Embedding
        # This allows the model to shift its interpretation based on geography
        self.eco_embedding = nn.Embedding(num_ecoregions, emb_dim)
        
        # 3. Global Feature Dimension (Backbone Latent + Embedding)
        latent_dim = 256 + emb_dim
        
        # --- PRETEXT HEADS ---
        # Task 1: Dominant Species (Weakly Supervised)
        self.pretext_species_head = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_species)
        )
        
        # Task 2: Structural Proxy (H95 Regression)
        self.pretext_structure_head = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1) # Outputs H95 in meters
        )
        
        # --- DOWNSTREAM HEAD ---
        # Task 3: Species Composition (FRI Expertise)
        self.downstream_comp_head = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_species),
            nn.Softmax(dim=-1) # Ensures sum-to-one for composition %
        )

    def forward(self, data, mode='pretext'):
        """
        data: dictionary from Dataset __getitem__
        mode: 'pretext' or 'downstream'
        """
        # data['pos']: (B, N, 3) - meters for grouping
        # data['x']:   (B, 3, N) - standardized for features (PointNeXt expects 3, N)
        # data['ecoregion']: (B,)
        
        # 1. Get Backbone Latent Features
        # PointNeXt expects features in (B, C, N) format
        x = data['x'].transpose(1, 2) if data['x'].shape[1] != 3 else data['x']
        global_feat = self.backbone.encoder(data['pos'], x)
        global_feat = self.backbone.prediction(global_feat) # (B, 256)
        
        # 2. Add Context (Ecoregion Embedding)
        eco_emb = self.eco_embedding(data['ecoregion']) # (B, emb_dim)
        combined_feat = torch.cat([global_feat, eco_emb], dim=-1) # (B, 256 + emb_dim)
        
        if mode == 'pretext':
            species_logits = self.pretext_species_head(combined_feat)
            h95_pred = self.pretext_structure_head(combined_feat)
            return {
                "species_logits": species_logits,
                "h95_pred": h95_pred.squeeze(-1)
            }
        
        elif mode == 'downstream':
            comp_pred = self.downstream_comp_head(combined_feat)
            return {
                "composition_pred": comp_pred
            }