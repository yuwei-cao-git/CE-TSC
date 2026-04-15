import torch
import torch.nn as nn
import torch.nn.functional as F
from pointnext import pointnext_s, PointNext, pointnext_b, pointnext_l, pointnext_xl

class LogitAlignmentHead(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features=in_features, out_features=out_features, bias=bias)
        self.confidence_layer = nn.Linear(in_features, 1)
        self.logit_scale = nn.Parameter(torch.ones(1, out_features))
        self.logit_bias = nn.Parameter(torch.zeros(1, out_features))
        nn.init.constant_(self.confidence_layer.weight, 0.1)

    def forward(self, input):
        logit_before = F.linear(input, self.weight, self.bias)
        confidence = self.confidence_layer(input).sigmoid()
        logit_after = (
            1 + confidence * self.logit_scale
        ) * logit_before + confidence * self.logit_bias
        return logit_after

class PointNextOntario(nn.Module):
    def __init__(self, config, in_dim, num_species, num_ecoregions):
        super(PointNextOntario, self).__init__()
        self.config = config

        # 1. Select Encoder from PyPI package
        if config.get("encoder", "s") == "s":
            encoder_backbone = pointnext_s(in_dim=in_dim)
        elif config["encoder"] == "b":
            encoder_backbone = pointnext_b(in_dim=in_dim)
        elif config["encoder"] == "l":
            encoder_backbone = pointnext_l(in_dim=in_dim)
        else:
            encoder_backbone = pointnext_xl(in_dim=in_dim)

        # 2. Backbone Wrapper
        # emb_dims is the bottleneck dimension (usually 512 for S, 1024 for L)
        self.backbone = PointNext(config["emb_dims"], encoder=encoder_backbone)
        in_dims = 2 * config["emb_dims"]
        # self.bn_out = nn.BatchNorm1d(config["emb_dims"])
        self.out_norm = nn.LayerNorm(in_dims)
        self.act = nn.ReLU()

        # 3. Context: Ecoregion Embedding
        # Allows model to interpret structure differently based on geography
        if config["eco_emb_dim"] > 0:
            self.eco_embedding = nn.Embedding(num_ecoregions, config["eco_emb_dim"])
            in_dims += config["eco_emb_dim"]
        if config["align_head"]:
            self.disalign_head = LogitAlignmentHead(num_species, num_species)

        # --- PRETEXT HEADS & DOWNSTREAM HEAD ---
        if config["mode"] in ["pretext_lsc", "pretext_both"]:
            # Task A: Species Classification (Weak Supervision)
            self.species_head = nn.Sequential(
                nn.Linear(in_dims, 512),
                nn.LayerNorm(512),
                nn.GELU(),
                nn.Dropout(0.3),
                nn.Linear(512, 256),
                nn.LayerNorm(256),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(256, num_species),
            )
        if config["mode"] == "pretext_both":
            # Task B: Structural Regression (H95 Prediction)
            """ self.structure_head = nn.Sequential(
                nn.Linear(config["emb_dims"] + config["eco_emb_dim"], 128),
                nn.ReLU(),
                nn.Linear(128, 1),  # Raw meters (float)
            ) """
            # try a complex head
            self.structure_head = nn.Sequential(
                nn.Linear(in_dims, 256),
                nn.LayerNorm(256),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(256, 128),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(128, 1),
                nn.Softplus(),
            )

        if config["mode"] == "downstream":
            # Task C: Species Composition Regression
            self.composition_head = nn.Sequential(
                nn.Linear(in_dims, 512),
                nn.LayerNorm(512),
                nn.GELU(),
                nn.Dropout(0.3),
                nn.Linear(512, 256),
                nn.LayerNorm(256),
                nn.GELU(),
                nn.Linear(256, num_species),
            )

    def forward(self, pc_feat, xyz, eco_idx, mode=None):
        """
        pc_feat: (B, 3, N) Standardized coordinates as features
        xyz:     (B, 3, N) Centered coordinates for grouping
        eco_idx: (B,) Integer IDs for ecoregions
        mode:    'pretext_lsc' 'pretext_both' or 'downstream'
        """
        # 1. Feature Extraction
        global_features = self.backbone(pc_feat, xyz)  # Output: (B, emb_dims, N)
        global_avg = global_features.mean(dim=-1)  # (B, emb_dims)
        global_max = global_features.max(dim=-1)[0]
        out = torch.cat([global_avg, global_max], dim=-1)

        out = self.out_norm(out)
        out = self.act(out)

        # 2. Inject Ecoregion Context
        if eco_idx is not None and self.config.get("eco_emb_dim", 0) > 0:
            eco_feat = self.eco_embedding(eco_idx)  # (B, eco_emb_dim)
            out = torch.cat([out, eco_feat], dim=-1)  # (B, latent_dim)

        # 3. Branching Logic
        if mode == "pretext_lsc":
            species_logits = self.species_head(out)
            if self.config["align_head"]:
                species_logits = self.disalign_head(species_logits)
            return species_logits
        elif mode == "pretext_both":
            species_logits = self.species_head(out)
            if self.config["align_head"]:
                species_logits = self.disalign_head(species_logits)
            h95_pred = self.structure_head(out).squeeze(-1)
            return species_logits, h95_pred

        elif mode == "downstream":
            logits = self.composition_head(out)
            if self.config["align_head"]:
                logits = self.disalign_head(logits)
            comp_pred = F.softmax(logits, dim=1)
            return comp_pred
        else:
            raise ValueError(f"Unknown mode: {mode}")


class ResidualSpeciesHead(nn.Module):
    def __init__(self, in_dim, num_species):
        super().__init__()

        self.fc1 = nn.Linear(in_dim, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 256)
        self.out = nn.Linear(256, num_species)

        self.norm1 = nn.LayerNorm(512)
        self.norm2 = nn.LayerNorm(512)

    def forward(self, x):
        x = F.gelu(self.norm1(self.fc1(x)))

        res = x
        x = F.gelu(self.norm2(self.fc2(x)))
        x = x + res

        x = F.gelu(self.fc3(x))
        return self.out(x)
