# =============================================================================
# models_addendum.py  --  Baselines that receive the node encoding.
#
# Each model here is the counterpart of one in models.py, differing only in
# that the per-location encoding U is projected and added to the
# representation. Keeping the injection identical across architectures is
# what lets a comparison between them isolate the representation rather than
# the encoding.
#
# Where the encoding enters differs by architecture, and that is a property
# of the architecture rather than a choice: SimpleViT_U and
# VideoTransformer_U add it to every token, while ResNet18_U and CNN_LSTM_U
# have only a pooled vector to add it to. GATWithPE and GCNWithPE
# concatenate it into the node features before aggregation.
#
# Forward interface for the graph models, matching fwd_gat_pe in main.py:
#   forward(x, x_nbr, nbr_mask, u_q, u_nbr)
# =============================================================================

import torch
import torch.nn as nn

from models import (_CNNEncoder, GATConv, ResNet18,
                    VideoTransformer)


class ResNet18_U(nn.Module):
    """
    ResNet-18 with the physical node encoding injected additively.

    This is the no-graph control for the node-encoding experiments: it
    receives exactly the same per-location encoding U as the graph models,
    projected and added to the pooled image feature, but performs no
    neighbour aggregation. The gap between this and a graph model using the
    identical encoding therefore isolates the value of aggregation itself,
    rather than confounding it with what the encoding contributes.

    Injection matches SpatiotemporalViT: a linear projection of U added to
    the representation (not concatenated, not used as a gate), so that the
    comparison differs in one factor only.
    """

    def __init__(self, channels=1, num_classes=1, base_ch=8, dropout=0.0,
                 u_dim=30):
        super().__init__()
        c = base_ch
        self.stem = nn.Sequential(
            nn.Conv2d(channels, c, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
        )
        self.layer1 = ResNet18._make_stage(c,     c,     2, stride=1)
        self.layer2 = ResNet18._make_stage(c,     c * 2, 2, stride=2)
        self.layer3 = ResNet18._make_stage(c * 2, c * 4, 2, stride=2)
        self.layer4 = ResNet18._make_stage(c * 4, c * 8, 2, stride=2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.u_proj = nn.Linear(u_dim, c * 8)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(c * 8, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, U):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        feat = self.pool(x).flatten(1)                      # [B, c*8]
        feat = feat + self.u_proj(U.to(feat.device, feat.dtype))
        return self.head(self.drop(feat))


class CNN_LSTM_U(nn.Module):
    """
    CNN+LSTM with the physical node encoding injected additively.

    The encoding describes the query location, not individual frames, so it
    is added once to the pooled sequence representation -- the same place
    ResNet18_U adds it, and for the same reason: this is where the model's
    single per-location descriptor lives.
    """

    def __init__(self, channels=1, feat_dim=256, hidden_dim=128,
                 num_layers=2, num_classes=1, dropout=0.1, u_dim=30):
        super().__init__()
        self.encoder = _CNNEncoder(channels=channels, feat_dim=feat_dim)
        self.lstm = nn.LSTM(input_size=feat_dim, hidden_size=hidden_dim,
                            num_layers=num_layers, batch_first=True,
                            bidirectional=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.u_proj = nn.Linear(u_dim, hidden_dim * 2)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_dim * 2, num_classes)
        for name, p in self.lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(p)
            elif "weight_hh" in name:
                nn.init.orthogonal_(p)
            elif "bias" in name:
                nn.init.zeros_(p)
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        nn.init.xavier_uniform_(self.u_proj.weight)
        nn.init.zeros_(self.u_proj.bias)

    def forward(self, x, U):                       # x: [B, T, C, H, W]
        B, T, C, H, W = x.shape
        feats = self.encoder(x.reshape(B * T, C, H, W)).reshape(B, T, -1)
        _, (hn, _) = self.lstm(feats)
        z = torch.cat([hn[-2], hn[-1]], dim=-1)     # [B, 2*hidden]
        z = z + self.u_proj(U.to(z.device, z.dtype))
        return self.head(self.drop(z))


class VideoTransformer_U(nn.Module):
    """
    VideoTransformer (ViViT-style) with the physical node encoding added to
    every spatiotemporal token, matching how SimpleViT_U and STGT inject it.
    """

    def __init__(self, *, image_size=120, patch_size=24, seq_len=11,
                 channels=1, dim=128, depth=4, heads=4, dim_head=64,
                 mlp_dim=256, dropout=0.1, num_classes=1, u_dim=30):
        super().__init__()
        self.backbone = VideoTransformer(
            image_size=image_size, patch_size=patch_size, seq_len=seq_len,
            channels=channels, dim=dim, depth=depth, heads=heads,
            dim_head=dim_head, mlp_dim=mlp_dim, dropout=dropout,
            num_classes=num_classes)
        self.u_proj = nn.Linear(u_dim, dim)
        nn.init.xavier_uniform_(self.u_proj.weight)
        nn.init.zeros_(self.u_proj.bias)

    def forward(self, x, U):                       # x: [B, T, C, H, W]
        b = self.backbone
        z = b._embed(x)                             # [B, T*N, dim]
        g = self.u_proj(U.to(z.device, z.dtype)).unsqueeze(1)
        z = z + g                                   # broadcast over tokens
        for attn, ffn in b.blocks:
            z = z + attn(z)
            z = z + ffn(z)
        return b.head(b.final_norm(z).mean(dim=1))


class GCNConv(nn.Module):
    """
    Graph convolution with fixed aggregation weights (Kipf & Welling 2017),
    the convolutional counterpart to GATConv.

    Where GAT learns a coefficient for each neighbour from node features,
    GCN uses a coefficient determined solely by the graph:

        h_i' = W ( sum_j  1/sqrt(d_i d_j)  h_j )

    Including this baseline separates two things that GAT confounds: the
    value of aggregating over a neighbourhood at all, and the value of
    weighting that aggregation by learned attention. Degrees are computed
    from the causal mask, so they reflect the neighbours actually visible.
    """

    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.lin = nn.Linear(dim, dim, bias=False)
        self.drop = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim, bias=False)
        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.xavier_uniform_(self.proj.weight)

    def forward(self, x_q, x_kv, nbr_mask):
        """
        x_q     : [B, dim]
        x_kv    : [B, K, dim]
        nbr_mask: [B, K] bool -- True where the neighbour is valid
        """
        h_kv = self.lin(self.norm_kv(x_kv))                  # [B, K, dim]
        m = nbr_mask.to(h_kv.dtype).unsqueeze(-1)            # [B, K, 1]
        deg = m.sum(dim=1).clamp(min=1.0)                    # [B, 1]
        # symmetric normalization degenerates to 1/deg here because the
        # neighbour table is of fixed width; use mean over valid slots
        out = (h_kv * m).sum(dim=1) / deg                    # [B, dim]
        return self.proj(self.drop(out))


class GCNWithPE(nn.Module):
    """GCN baseline with the same node encoding and block structure as GAT."""

    def __init__(self, *, channels=1, feat_dim=256, u_dim=30,
                 gcn_dim=128, depth=4, num_classes=1, dropout=0.1):
        super().__init__()
        self.encoder = _CNNEncoder(channels=channels, feat_dim=feat_dim)
        self.input_proj = nn.Sequential(
            nn.Linear(feat_dim + u_dim, gcn_dim),
            nn.LayerNorm(gcn_dim),
        )
        self.blocks = nn.ModuleList([
            nn.ModuleList([
                GCNConv(gcn_dim, dropout=dropout),
                nn.Sequential(
                    nn.LayerNorm(gcn_dim),
                    nn.Linear(gcn_dim, gcn_dim * 2),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(gcn_dim * 2, gcn_dim),
                ),
            ])
            for _ in range(depth)
        ])
        self.final_norm = nn.LayerNorm(gcn_dim)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(gcn_dim, num_classes)
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def _encode(self, img, u):
        *lead, C, H, W = img.shape
        feat = self.encoder(img.reshape(-1, C, H, W)).reshape(*lead, -1)
        return self.input_proj(torch.cat([feat, u], dim=-1))

    def forward(self, x, x_nbr, nbr_mask, u_q, u_nbr):
        h_q = self._encode(x, u_q)
        h_kv = self._encode(x_nbr, u_nbr)
        for conv, ffn in self.blocks:
            h_q = h_q + conv(h_q, h_kv, nbr_mask)
            h_q = h_q + ffn(h_q)
        return self.head(self.drop(self.final_norm(h_q)))


class GATWithPE(nn.Module):
    """GAT baseline + node PE concatenated into the input projection."""

    def __init__(self, *, channels=1, feat_dim=256, u_dim=30,
                 gat_dim=128, heads=4, depth=4,
                 num_classes=1, dropout=0.1):
        super().__init__()
        self.encoder = _CNNEncoder(channels=channels, feat_dim=feat_dim)
        self.input_proj = nn.Sequential(
            nn.Linear(feat_dim + u_dim, gat_dim),
            nn.LayerNorm(gat_dim),
        )
        self.blocks = nn.ModuleList([
            nn.ModuleList([
                GATConv(gat_dim, heads=heads, dropout=dropout),
                nn.Sequential(
                    nn.LayerNorm(gat_dim),
                    nn.Linear(gat_dim, gat_dim * 2),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(gat_dim * 2, gat_dim),
                ),
            ])
            for _ in range(depth)
        ])
        self.final_norm = nn.LayerNorm(gat_dim)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(gat_dim, num_classes)
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def _encode(self, img, u):
        *lead, C, H, W = img.shape
        feat = self.encoder(img.reshape(-1, C, H, W)).reshape(*lead, -1)
        return self.input_proj(torch.cat([feat, u], dim=-1))

    def forward(self, x, x_nbr, nbr_mask, u_q, u_nbr):
        h_q = self._encode(x, u_q)                    # [B, gat_dim]
        h_kv = self._encode(x_nbr, u_nbr)             # [B, K, gat_dim]
        for gat_conv, ffn in self.blocks:
            h_q = h_q + gat_conv(h_q, h_kv, nbr_mask)
            h_q = h_q + ffn(h_q)
        return self.head(self.drop(self.final_norm(h_q)))
