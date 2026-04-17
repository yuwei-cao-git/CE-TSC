import torch
import torch.nn as nn
import torch.nn.functional as F
from pointnext import pointnext_s, PointNext, pointnext_b, pointnext_l

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


class CompositionMTLHead(nn.Module):
    def __init__(self, in_dim, num_species, align=False):
        super().__init__()

        self.shared = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
        )

        self.dom_head = nn.Linear(256, num_species)

        # self.comp_head = nn.Linear(256 + num_species, num_species)

        self.residual_head = nn.Sequential(
            nn.Linear(256 + num_species, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, num_species),
        )

    def forward(self, x):
        feat = self.shared(x)

        dom_logits = self.dom_head(feat)
        dom_prob = F.softmax(dom_logits, dim=1)

        residual_logits = self.residual_head(torch.cat([feat, dom_prob], dim=1))
        comp_logits = dom_logits + residual_logits

        return dom_logits, comp_logits


class PointNextOntario(nn.Module):
    def __init__(self, config, in_dim, num_species, num_ecoregions):
        super(PointNextOntario, self).__init__()
        self.config = config

        # 1. Select Encoder from PyPI package
        encoder_map = {
            "s": pointnext_s,
            "b": pointnext_b,
            "l": pointnext_l,
        }
        encoder_fn = encoder_map.get(config.get("encoder", "s"), pointnext_s)
        encoder_backbone = encoder_fn(in_dim=in_dim)

        # 2. Backbone Wrapper
        # emb_dims is the bottleneck dimension (usually 512 for S, 1024 for L)
        self.backbone = PointNext(config["emb_dims"], encoder=encoder_backbone)
        in_dims = 2 * config["emb_dims"]

        # self.bn_out = nn.BatchNorm1d(config["emb_dims"])
        self.out_norm = nn.LayerNorm(in_dims)
        self.act = nn.GELU()

        # 3. Context: Ecoregion Embedding
        # Allows model to interpret structure differently based on geography
        if config.get("eco_emb_dim", 0) > 0:
            self.eco_embedding = nn.Embedding(num_ecoregions, config["eco_emb_dim"])
            in_dims += config["eco_emb_dim"]

        # --- Image Embedding Handling ---
        if "embedding" in config.get("mode", ""):
            # We add a projection layer to blend the 128-dim foundation
            # features into our model's feature space
            self.img_projection = nn.Sequential(
                nn.Linear(128, 512), nn.LayerNorm(512), nn.GELU(), nn.Dropout(0.1)
            )
            in_dims += 512  # Final head input will be 1536 + 512 = 2048

        if config.get("align_head", False):
            self.disalign_head = LogitAlignmentHead(num_species, num_species)

        # --- PRETEXT HEADS & DOWNSTREAM HEAD ---
        mode = config.get("mode", "pretext_both")
        if "pretext" in mode:
            self.species_head = nn.Sequential(
                nn.Linear(in_dims, 512),
                nn.LayerNorm(512),
                nn.GELU(),
                nn.Dropout(0.3),
                nn.Linear(512, 256),
                nn.LayerNorm(256),
                nn.GELU(),
                nn.Linear(256, num_species),
            )
            if mode == "pretext_both":
                self.structure_head = nn.Sequential(
                    nn.Linear(in_dims, 256),
                    nn.LayerNorm(256),
                    nn.GELU(),
                    nn.Linear(256, 128),
                    nn.GELU(),
                    nn.Linear(128, 1),
                    nn.Softplus(),  # Ensures height is always positive
                )

        elif "downstream" in mode:
            if mode in ["downstream", "downstream_embedding"]:
                # Standard Composition Prediction
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
            elif mode in ["downstream_both"]:
                # Multi-task: Leading + Composition
                self.mtl_head = CompositionMTLHead(in_dims, num_species)

    def forward(self, pc_feat, xyz, eco_idx, patch_embed=None, mode=None):
        """
        pc_feat: (B, 3, N) Standardized coordinates as features
        xyz:     (B, 3, N) Centered coordinates for grouping
        eco_idx: (B,) Integer IDs for ecoregions
        mode:    'pretext_lsc' 'pretext_both' or 'downstream'
        """
        # --- 1. Encoder & Multi-Pooling ---
        # Input: pc_feat (B, 3, N), xyz (B, 3, N)
        point_features = self.backbone(pc_feat, xyz)

        avg_pool = point_features.mean(dim=-1)
        max_pool = point_features.max(dim=-1)[0]
        out = torch.cat([avg_pool, max_pool], dim=-1)  # (B, 2*emb_dims)
        out = self.out_norm(out)
        out = self.act(out)

        # --- 2. Context Injection ---
        if eco_idx is not None and self.config.get("eco_emb_dim", 0) > 0:
            eco_feat = self.eco_embedding(eco_idx)
            out = torch.cat([out, eco_feat], dim=-1)
        # 2. Image Feature Injection
        if patch_embed is not None and hasattr(self, "img_projection"):
            # patch_embed is (B, 3, 3, 128)
            # Pool spatial grid: (B, 3, 3, 128) -> (B, 128)
            img_vec = patch_embed.mean(dim=(1, 2))
            img_feat = self.img_projection(img_vec)  # (B, 512)
            out = torch.cat([out, img_feat], dim=-1)  # (B, 2048)

        # --- 3. Branching Logic ---
        if mode == "pretext_lsc":
            logits = self.species_head(out)
            if hasattr(self, "disalign_head"):
                logits = self.disalign_head(logits)
            return logits

        elif mode == "pretext_both":
            logits = self.species_head(out)
            if hasattr(self, "disalign_head"):
                logits = self.disalign_head(logits)
            h95 = self.structure_head(out).squeeze(-1)
            return logits, h95

        elif mode in ["downstream", "downstream_embedding"]:
            logits = self.composition_head(out)
            if hasattr(self, "disalign_head"):
                logits = self.disalign_head(logits)
            # Returning Softmax for composition as it's a proportion task
            return F.softmax(logits, dim=1)

        elif mode in ["downstream_both", "downstream_both_embedding"]:
            dom_logits, comp_logits = self.mtl_head(out)
            if hasattr(self, "disalign_head"):
                dom_logits = self.disalign_head(dom_logits)
                comp_logits = self.disalign_head(comp_logits)
            # Leading species: Logits | Composition: Softmax
            return dom_logits, F.softmax(comp_logits, dim=1)

        else:
            raise ValueError(f"Unknown mode: {mode}")
