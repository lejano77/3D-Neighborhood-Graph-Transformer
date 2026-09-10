# =============================================================================
# models.py  --  All model classes
#
# Contents
# --------
# Utilities
#   pair, posemb_sincos_1d, posemb_sincos_2d
#
# Shared primitives
#   FeedForward, Attention, CrossAttention
#
# Transformer stacks
#   Transformer, TransformerCross
#
# Image-only models
#   SimpleViT, SimpleViT_U
#   _BasicBlock, ResNet18
#
# Sequence-based models
#   _CNNEncoder, CNN_LSTM
#   _VideoAttention, _VideoFeedForward, VideoTransformer
#
# Graph-based models
#   SpatiotemporalViT
#   _ImageCNNEncoder, GATConv, GAT
# =============================================================================

import torch
import torch.nn as nn
from einops import rearrange
from einops.layers.torch import Rearrange


# =============================================================================
# Utilities
# =============================================================================

def pair(t):
    """Return (t, t) if t is not already a tuple."""
    return t if isinstance(t, tuple) else (t, t)


def posemb_sincos_1d(n, dim, temperature=10000.0,
                     dtype=torch.float32, device=None):
    """Fixed 1-D sinusoidal positional embedding.  Returns [n, dim]."""
    assert dim % 2 == 0, "dim must be even for 1-D sincos"
    half  = dim // 2
    omega = torch.arange(half, device=device, dtype=dtype) / max(half - 1, 1)
    omega = 1.0 / (temperature ** omega)
    pos   = torch.arange(n, device=device, dtype=dtype)
    out   = torch.outer(pos, omega)                              # [n, half]
    return torch.cat([out.sin(), out.cos()], dim=-1)             # [n, dim]


def posemb_sincos_2d(h, w, dim, temperature=10000.0,
                     dtype=torch.float32, device=None):
    """Fixed 2-D sinusoidal positional embedding.  Returns [h*w, dim]."""
    assert dim % 4 == 0 and dim >= 8, "dim must be divisible by 4 and >= 8"
    d        = dim // 4
    inv_freq = 1.0 / (temperature ** (
        torch.arange(d, device=device, dtype=dtype) / max(d - 1, 1)
    ))
    y  = torch.arange(h, device=device, dtype=dtype)
    x  = torch.arange(w, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    yy = yy.reshape(-1, 1) * inv_freq.reshape(1, -1)            # [h*w, d]
    xx = xx.reshape(-1, 1) * inv_freq.reshape(1, -1)            # [h*w, d]
    pe = torch.cat([xx.sin(), xx.cos(), yy.sin(), yy.cos()], dim=1)
    return pe.to(dtype=dtype, device=device)                     # [h*w, dim]


# =============================================================================
# Shared primitives  (used by ViT, TransformerCross, VideoTransformer)
# =============================================================================

class FeedForward(nn.Module):
    """MLP block with pre-LayerNorm.  x: [B, N, D] -> [B, N, D]"""
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1  = nn.Linear(dim, hidden_dim)
        self.act  = nn.GELU()
        self.fc2  = nn.Linear(hidden_dim, dim)

    def forward(self, x):
        xn = self.norm(x)
        return self.fc2(self.act(self.fc1(xn)))


class Attention(nn.Module):
    """Multi-head self-attention with pre-LayerNorm.  x: [B, N, D] -> [B, N, D]"""
    def __init__(self, dim, heads=8, dim_head=64):
        super().__init__()
        inner      = heads * dim_head
        self.h     = heads
        self.scale = dim_head ** -0.5
        self.norm  = nn.LayerNorm(dim)
        self.qkv   = nn.Linear(dim, inner * 3, bias=False)
        self.out   = nn.Linear(inner, dim, bias=False)

    def forward(self, x):
        xn      = self.norm(x)
        q, k, v = self.qkv(xn).chunk(3, dim=-1)
        q = rearrange(q, 'b n (h d) -> b h n d', h=self.h)
        k = rearrange(k, 'b n (h d) -> b h n d', h=self.h)
        v = rearrange(v, 'b n (h d) -> b h n d', h=self.h)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out  = attn @ v
        out  = rearrange(out, 'b h n d -> b n (h d)')
        return self.out(out)


class CrossAttention(nn.Module):
    """
    Multi-head cross-attention with pre-LayerNorm.
    query = current node tokens  [B, N, D]
    key/value = neighbor tokens  [B, M, D]

    cross_mask [B, N, M] bool -- True where attention is allowed.
    Fully-masked rows (isolated / all-future-layer nodes) are guarded
    against NaN via nan_to_num after softmax.
    """
    def __init__(self, dim, heads=4, dim_head=64):
        super().__init__()
        inner        = heads * dim_head
        self.h       = heads
        self.scale   = dim_head ** -0.5
        self.norm_q  = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.to_q    = nn.Linear(dim, inner, bias=False)
        self.to_kv   = nn.Linear(dim, inner * 2, bias=False)
        self.out     = nn.Linear(inner, dim, bias=False)
    
    def forward(self, x_q, x_kv, cross_mask=None, return_attn=False):
        x_q  = self.norm_q(x_q)
        x_kv = self.norm_kv(x_kv)
        q       = self.to_q(x_q)
        k, v    = self.to_kv(x_kv).chunk(2, dim=-1)
        q = rearrange(q, 'b n (h d) -> b h n d', h=self.h)
        k = rearrange(k, 'b m (h d) -> b h m d', h=self.h)
        v = rearrange(v, 'b m (h d) -> b h m d', h=self.h)
        attn = (q @ k.transpose(-2, -1)) * self.scale       # [B, H, N, M]
        if cross_mask is not None:
            m    = cross_mask[:, None, :, :].to(torch.bool)
            attn = attn.masked_fill(~m, torch.finfo(attn.dtype).min)
        attn = attn.softmax(dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)
        out  = attn @ v
        out  = rearrange(out, 'b h n d -> b n (h d)')
        out  = self.out(out)
        if return_attn:
            return out, attn                                 # [B, H, N_q, N_kv]
        return out
    

# =============================================================================
# Transformer stacks
# =============================================================================

class Transformer(nn.Module):
    """Standard self-attention transformer stack (no cross-attention)."""
    def __init__(self, dim, depth, heads, dim_head, mlp_dim):
        super().__init__()
        self.blocks = nn.ModuleList([
            nn.ModuleList([
                Attention(dim, heads, dim_head),
                FeedForward(dim, mlp_dim),
            ])
            for _ in range(depth)
        ])
        self.final_norm = nn.LayerNorm(dim)

    def forward(self, x):
        for attn, ffn in self.blocks:
            x = x + attn(x)
            x = x + ffn(x)
        return self.final_norm(x)


class TransformerCross(nn.Module):
    """
    Dual-attention transformer stack.

    Per block:
        Self-Attn -> FFN1 -> [spectral pos inject, once] -> Cross-Attn -> FFN2

    Spectral positional embeddings (u_q_tok, u_nbr_tok) injected once --
    after first self-attn has refined node representations, before
    cross-attn communicates with neighbors.

    FFN2 gated with cross-attn: only runs when x_nbr is provided,
    keeping both code paths symmetric in depth.
    """
    def __init__(self, dim, depth=4, heads=4, dim_head=64, mlp_dim=128):
        super().__init__()
        self.blocks = nn.ModuleList([
            nn.ModuleList([
                Attention(dim, heads, dim_head),       # within-node self-attn
                FeedForward(dim, mlp_dim),             # FFN after self-attn
                CrossAttention(dim, heads, dim_head),  # neighbor cross-attn
                FeedForward(dim, mlp_dim),             # FFN after cross-attn
            ])
            for _ in range(depth)
        ])
        self.final_norm = nn.LayerNorm(dim)
    
    def forward(self, x, x_nbr=None, cross_mask=None,
            u_q_tok=None, u_nbr_tok=None, return_attn=False):
        pos_injected = False
        all_attn     = []                                    # collect per-block attn
    
        for self_attn, ffn1, cross_attn, ffn2 in self.blocks:
            x = x + self_attn(x)
            x = x + ffn1(x)
    
            if not pos_injected:
                if u_q_tok is not None:
                    x = x + u_q_tok
                if x_nbr is not None and u_nbr_tok is not None:
                    x_nbr = x_nbr + u_nbr_tok
                pos_injected = True
    
            if x_nbr is not None:
                if return_attn:
                    ca_out, attn_w = cross_attn(
                        x, x_nbr, cross_mask=cross_mask, return_attn=True
                    )
                    all_attn.append(attn_w)                  # [B, H, N, K*N]
                    x = x + ca_out
                else:
                    x = x + cross_attn(x, x_nbr, cross_mask=cross_mask)
                x = x + ffn2(x)
    
        out = self.final_norm(x)
        if return_attn:
            return out, all_attn                             # list of [B,H,N,K*N]
        return out
    

# =============================================================================
# Image-only models
# =============================================================================

class SimpleViT(nn.Module):
    """Plain ViT for single-image regression."""
    def __init__(self, *, image_size=120, patch_size=12, num_classes=1,
                 dim=128, depth=4, heads=4, mlp_dim=256,
                 channels=1, dim_head=64):
        super().__init__()
        H,  W  = pair(image_size)
        Ph, Pw = pair(patch_size)
        assert H % Ph == 0 and W % Pw == 0, "image must be divisible by patch size"
        Pdim   = channels * Ph * Pw
        gh, gw = H // Ph, W // Pw

        self.to_patch_embedding = nn.Sequential(
            Rearrange("b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1=Ph, p2=Pw),
            nn.LayerNorm(Pdim),
            nn.Linear(Pdim, dim, bias=True),
            nn.LayerNorm(dim),
        )
        self.register_buffer(
            'pos_embedding', posemb_sincos_2d(gh, gw, dim).unsqueeze(0)
        )
        self.encoder = Transformer(dim, depth, heads, dim_head, mlp_dim)
        self.head    = nn.Linear(dim, num_classes)

    def forward(self, img):
        x = self.to_patch_embedding(img)
        x = x + self.pos_embedding.to(img.device, img.dtype)
        x = self.encoder(x)
        return self.head(x.mean(dim=1))

    @torch.no_grad()
    def extract_features(self, img):
        x = self.to_patch_embedding(img)
        x = x + self.pos_embedding.to(img.device, img.dtype)
        return self.encoder(x).mean(dim=1)


class SimpleViT_U(nn.Module):
    """ViT with graph spectral embedding U injected as a token offset."""
    def __init__(self, *, image_size=120, patch_size=12, num_classes=1,
                 dim=128, depth=4, heads=4, mlp_dim=256,
                 channels=1, dim_head=64, u_dim=1):
        super().__init__()
        H,  W  = pair(image_size)
        Ph, Pw = pair(patch_size)
        assert H % Ph == 0 and W % Pw == 0, "image must be divisible by patch size"
        Pdim   = channels * Ph * Pw
        gh, gw = H // Ph, W // Pw

        self.to_patch_embedding = nn.Sequential(
            Rearrange("b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1=Ph, p2=Pw),
            nn.LayerNorm(Pdim),
            nn.Linear(Pdim, dim, bias=True),
            nn.LayerNorm(dim),
        )
        self.register_buffer(
            'pos_embedding', posemb_sincos_2d(gh, gw, dim).unsqueeze(0)
        )
        self.u_proj  = nn.Linear(u_dim, dim)
        self.encoder = Transformer(dim, depth, heads, dim_head, mlp_dim)
        self.head    = nn.Linear(dim, num_classes)

    def forward(self, img, U):
        assert U.dim() == 2 and U.size(1) == self.u_proj.in_features, \
            f"U must be [B, {self.u_proj.in_features}], got {tuple(U.shape)}"
        T = self.to_patch_embedding(img)
        P = self.pos_embedding.to(img.device, img.dtype)
        G = self.u_proj(U.to(img.device, img.dtype))
        G = G.unsqueeze(1).expand(-1, T.size(1), -1)
        x = self.encoder(T + P + G).mean(dim=1)
        return self.head(x)


class _BasicBlock(nn.Module):
    """Standard ResNet BasicBlock with optional projection shortcut."""
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1    = nn.Conv2d(in_ch, out_ch, 3, stride=stride,
                                  padding=1, bias=False)
        self.bn1      = nn.BatchNorm2d(out_ch)
        self.conv2    = nn.Conv2d(out_ch, out_ch, 3, stride=1,
                                  padding=1, bias=False)
        self.bn2      = nn.BatchNorm2d(out_ch)
        self.act      = nn.ReLU(inplace=True)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x):
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.act(out + self.shortcut(x))


class ResNet18(nn.Module):
    """
    ResNet-18 adapted for grayscale regression.

    Stem   : 7x7 conv stride-2 -> BN -> ReLU -> 3x3 maxpool stride-2
    Layer1 : 2 x BasicBlock, base_ch,   stride 1
    Layer2 : 2 x BasicBlock, base_ch*2, stride 2
    Layer3 : 2 x BasicBlock, base_ch*4, stride 2
    Layer4 : 2 x BasicBlock, base_ch*8, stride 2
    Head   : GlobalAvgPool -> Dropout -> Linear(base_ch*8, num_classes)

    For 120x120 input: spatial dims after stem = 30x30,
    after all four layers = 3x3 before global pooling.

    Parameters
    ----------
    channels    : int   input channels (1 = grayscale)
    num_classes : int   output dim (1 = scalar regression)
    base_ch     : int   channel width; use 32 for half-width lighter variant
    dropout     : float dropout before head
    """
    def __init__(self, channels=1, num_classes=1, base_ch=64, dropout=0.0):
        super().__init__()
        c = base_ch
        self.stem = nn.Sequential(
            nn.Conv2d(channels, c, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
        )
        self.layer1 = self._make_stage(c,     c,     2, stride=1)
        self.layer2 = self._make_stage(c,     c * 2, 2, stride=2)
        self.layer3 = self._make_stage(c * 2, c * 4, 2, stride=2)
        self.layer4 = self._make_stage(c * 4, c * 8, 2, stride=2)
        self.pool   = nn.AdaptiveAvgPool2d(1)
        self.drop   = nn.Dropout(dropout)
        self.head   = nn.Linear(c * 8, num_classes)
        self._init_weights()

    @staticmethod
    def _make_stage(in_ch, out_ch, n_blocks, stride):
        layers = [_BasicBlock(in_ch, out_ch, stride=stride)]
        for _ in range(1, n_blocks):
            layers.append(_BasicBlock(out_ch, out_ch, stride=1))
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x); x = self.layer2(x)
        x = self.layer3(x); x = self.layer4(x)
        return self.head(self.drop(self.pool(x).flatten(1)))

    @torch.no_grad()
    def extract_features(self, x):
        x = self.stem(x)
        x = self.layer1(x); x = self.layer2(x)
        x = self.layer3(x); x = self.layer4(x)
        return self.pool(x).flatten(1)


# =============================================================================
# Sequence-based models
# =============================================================================

class _CNNEncoder(nn.Module):
    """
    Lightweight CNN: [B, C, H, W] -> [B, feat_dim].
    Four stride-2 conv blocks + global average pool.
    Shared by CNN_LSTM (per-frame encoding) and GAT (node image encoding).
    """
    def __init__(self, channels=1, feat_dim=256):
        super().__init__()
        dims = [channels, 32, 64, 128, feat_dim]
        layers = []
        for i in range(len(dims) - 1):
            layers += [
                nn.Conv2d(dims[i], dims[i + 1], 3,
                          stride=2, padding=1, bias=False),
                nn.BatchNorm2d(dims[i + 1]),
                nn.ReLU(inplace=True),
            ]
        self.net  = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        return self.pool(self.net(x)).flatten(1)


class CNN_LSTM(nn.Module):
    """
    CNN-per-frame encoder followed by a bidirectional LSTM.

    Forward input : x [B, T, C, H, W]  -- from SequenceDataset
    Output        : [B, num_classes]

    Architecture
    ------------
    1. _CNNEncoder per frame       -> [B, T, feat_dim]
    2. Bidirectional LSTM          -> hidden states
    3. Cat last fwd + bwd hidden   -> [B, 2*hidden_dim]
    4. Dropout -> Linear head      -> [B, num_classes]

    Parameters
    ----------
    channels    : int   image channels (1 = grayscale)
    feat_dim    : int   CNN output dim fed into LSTM
    hidden_dim  : int   LSTM hidden units per direction
    num_layers  : int   LSTM depth
    num_classes : int   output dim (1 = regression)
    dropout     : float between LSTM layers and before head
    """
    def __init__(self, channels=1, feat_dim=256, hidden_dim=128,
                 num_layers=2, num_classes=1, dropout=0.1):
        super().__init__()
        self.encoder = _CNNEncoder(channels=channels, feat_dim=feat_dim)
        self.lstm    = nn.LSTM(
            input_size=feat_dim, hidden_size=hidden_dim,
            num_layers=num_layers, batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_dim * 2, num_classes)
        self._init_weights()

    def _init_weights(self):
        for name, p in self.lstm.named_parameters():
            if   "weight_ih" in name: nn.init.xavier_uniform_(p)
            elif "weight_hh" in name: nn.init.orthogonal_(p)
            elif "bias"       in name: nn.init.zeros_(p)
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x):              # x: [B, T, C, H, W]
        B, T, C, H, W = x.shape
        feats = self.encoder(x.reshape(B * T, C, H, W)).reshape(B, T, -1)
        _, (hn, _) = self.lstm(feats)
        z = self.drop(torch.cat([hn[-2], hn[-1]], dim=-1))
        return self.head(z)

    @torch.no_grad()
    def extract_features(self, x):
        B, T, C, H, W = x.shape
        feats = self.encoder(x.reshape(B * T, C, H, W)).reshape(B, T, -1)
        _, (hn, _) = self.lstm(feats)
        return torch.cat([hn[-2], hn[-1]], dim=-1)


class _VideoAttention(nn.Module):
    """Multi-head self-attention with pre-LayerNorm (VideoTransformer block)."""
    def __init__(self, dim, heads, dim_head):
        super().__init__()
        inner      = heads * dim_head
        self.h     = heads
        self.scale = dim_head ** -0.5
        self.norm  = nn.LayerNorm(dim)
        self.qkv   = nn.Linear(dim, inner * 3, bias=False)
        self.out   = nn.Linear(inner, dim, bias=False)

    def forward(self, x):
        xn      = self.norm(x)
        q, k, v = self.qkv(xn).chunk(3, dim=-1)
        q = rearrange(q, 'b n (h d) -> b h n d', h=self.h)
        k = rearrange(k, 'b n (h d) -> b h n d', h=self.h)
        v = rearrange(v, 'b n (h d) -> b h n d', h=self.h)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        return self.out(rearrange(attn @ v, 'b h n d -> b n (h d)'))


class _VideoFeedForward(nn.Module):
    """MLP with pre-LayerNorm and dropout (VideoTransformer block)."""
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.net  = nn.Sequential(
            nn.Linear(dim, hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim), nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(self.norm(x))


class VideoTransformer(nn.Module):
    """
    Spatiotemporal Vision Transformer for frame sequences.

    Each frame is patchified independently. Spatial 2-D sincos PE and
    temporal 1-D sincos PE are added (additive decomposition), then all
    T*N tokens are processed jointly -- full space-time self-attention.

    No cross-attention with graph neighbors (contrast: SpatiotemporalViT).
    Temporal context comes only from frames within the fixed window.

    Forward input : x [B, T, C, H, W]  -- from SequenceDataset
    Output        : [B, num_classes]

    Parameters
    ----------
    image_size  : int or (h,w)
    patch_size  : int or (ph,pw)
    seq_len     : int           T, number of frames
    channels    : int           (1 = grayscale)
    dim         : int           token embedding dim
    depth       : int           transformer depth
    heads       : int           attention heads
    dim_head    : int           per-head dim
    mlp_dim     : int           FFN hidden dim
    dropout     : float
    num_classes : int           (1 = regression)
    """
    def __init__(self, *, image_size=120, patch_size=24, seq_len=8,
                 channels=1, dim=128, depth=4, heads=4, dim_head=64,
                 mlp_dim=256, dropout=0.0, num_classes=1):
        super().__init__()
        H, W   = pair(image_size)
        Ph, Pw = pair(patch_size)
        assert H % Ph == 0 and W % Pw == 0, "image must be divisible by patch size"
        self.seq_len     = seq_len
        self.num_patches = (H // Ph) * (W // Pw)
        self.dim         = dim
        patch_dim        = channels * Ph * Pw
        gh, gw           = H // Ph, W // Pw

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h ph) (w pw) -> b (h w) (ph pw c)', ph=Ph, pw=Pw),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )
        # spatial PE [1, 1, N, dim], temporal PE [1, T, 1, dim]
        self.register_buffer(
            'spatial_pe',
            posemb_sincos_2d(gh, gw, dim).unsqueeze(0).unsqueeze(0),
        )
        self.register_buffer(
            'temporal_pe',
            posemb_sincos_1d(seq_len, dim).unsqueeze(0).unsqueeze(2),
        )
        self.blocks = nn.ModuleList([
            nn.ModuleList([
                _VideoAttention(dim, heads, dim_head),
                _VideoFeedForward(dim, mlp_dim, dropout),
            ])
            for _ in range(depth)
        ])
        self.final_norm = nn.LayerNorm(dim)
        self.head       = nn.Linear(dim, num_classes)
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def _embed(self, x):
        """Patch embed + spatial PE + temporal PE -> [B, T*N, dim]."""
        B, T, C, H, W = x.shape
        frames = self.to_patch_embedding(
            x.reshape(B * T, C, H, W)
        ).reshape(B, T, self.num_patches, self.dim)
        frames = frames + self.spatial_pe.to(x.dtype)
        frames = frames + self.temporal_pe.to(x.dtype)
        return frames.reshape(B, T * self.num_patches, self.dim)

    def forward(self, x):
        z = self._embed(x)
        for attn, ffn in self.blocks:
            z = z + attn(z)
            z = z + ffn(z)
        return self.head(self.final_norm(z).mean(dim=1))

    @torch.no_grad()
    def extract_features(self, x):
        z = self._embed(x)
        for attn, ffn in self.blocks:
            z = z + attn(z)
            z = z + ffn(z)
        return self.final_norm(z).mean(dim=1)


# =============================================================================
# Graph-based models
# =============================================================================

class SpatiotemporalViT(nn.Module):
    """
    ViT with dual-attention (self + cross) and graph spectral embeddings.

    Forward inputs
    --------------
    x          : [B, C, H, W]       -- query image
    x_nbr      : [B, K, C, H, W]   -- K neighbor images
    cross_mask : [B, N, K*N]        -- True where cross-attn is allowed
                                       N = num_patches; causal: same/earlier layer
    u_q        : [B, u_dim]         -- spectral embedding of query node
    u_nbr      : [B, K, u_dim]     -- spectral embeddings of neighbor nodes

    Parameters
    ----------
    image_size  : int or (h,w)
    patch_size  : int or (ph,pw)
    num_classes : int
    dim         : int   token embedding dim
    depth       : int   transformer depth
    heads       : int   attention heads
    mlp_dim     : int   FFN hidden dim
    channels    : int   (1 = grayscale)
    dim_head    : int   per-head dim
    u_dim       : int   spectral embedding dim (k-1)
    """
    def __init__(self, *, image_size=120, patch_size=24, num_classes=1,
                 dim=128, depth=4, heads=4, mlp_dim=256,
                 channels=1, dim_head=64, u_dim=30):
        super().__init__()
        H,  W  = pair(image_size)
        Ph, Pw = pair(patch_size)
        assert H % Ph == 0 and W % Pw == 0, "image must be divisible by patch size"
        self.num_patches = (H // Ph) * (W // Pw)
        self.dim         = dim
        patch_dim        = channels * Ph * Pw
        gh, gw           = H // Ph, W // Pw

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h ph) (w pw) -> b (h w) (ph pw c)', ph=Ph, pw=Pw),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )
        self.register_buffer(
            'pos_embedding', posemb_sincos_2d(gh, gw, dim).unsqueeze(0)
        )
        self.u_proj      = nn.Linear(u_dim, dim)
        self.transformer = TransformerCross(dim, depth, heads, dim_head, mlp_dim)
        self.head        = nn.Linear(dim, num_classes)
        
    def forward(self, x, x_nbr, cross_mask, u_q, u_nbr,
                return_attn=False):
        B, C, H, W    = x.shape
        _, K, _, _, _ = x_nbr.shape
        N             = self.num_patches

        assert cross_mask.shape == (B, N, K * N), \
            f"cross_mask {tuple(cross_mask.shape)} != ({B}, {N}, {K * N})"

        # query tokens
        x = self.to_patch_embedding(x)
        x = x + self.pos_embedding.to(x.device, x.dtype)

        # neighbor tokens
        x_nbr     = x_nbr.reshape(B * K, C, H, W)
        x_nbr_tok = self.to_patch_embedding(x_nbr)
        x_nbr_tok = x_nbr_tok + self.pos_embedding.to(
            x_nbr_tok.device, x_nbr_tok.dtype
        )
        x_nbr_tok = x_nbr_tok.reshape(B, K * N, self.dim)

        # spectral embeddings
        u_q_tok   = self.u_proj(u_q.to(x.device, x.dtype))
        u_q_tok   = u_q_tok.unsqueeze(1).expand(-1, N, -1)

        _, K_u, Du = u_nbr.shape
        assert K_u == K
        u_nbr_tok = self.u_proj(
            u_nbr.reshape(B * K_u, Du).to(x.device, x.dtype)
        )
        u_nbr_tok = u_nbr_tok.unsqueeze(1).expand(-1, N, -1)
        u_nbr_tok = u_nbr_tok.reshape(B, K * N, self.dim)

        if return_attn:
            z, all_attn = self.transformer(
                x, x_nbr=x_nbr_tok, cross_mask=cross_mask,
                u_q_tok=u_q_tok, u_nbr_tok=u_nbr_tok,
                return_attn=True,
            )
            return self.head(z.mean(dim=1)), all_attn
        else:
            z = self.transformer(
                x, x_nbr=x_nbr_tok, cross_mask=cross_mask,
                u_q_tok=u_q_tok, u_nbr_tok=u_nbr_tok,
            )
            return self.head(z.mean(dim=1))
    

class GATConv(nn.Module):
    """
    Graph Attention Convolution (Velickovic et al. 2018), multi-head.

    Pre-LayerNorm. Query node attends over K neighbor nodes.
    Future-layer neighbors (nbr_mask=False) are blocked before softmax.
    Fully-masked rows produce zero output via nan_to_num.

    Parameters
    ----------
    dim     : int   node feature dim (input = output)
    heads   : int   attention heads (must divide dim)
    dropout : float attention dropout
    """
    def __init__(self, dim, heads=4, dropout=0.1):
        super().__init__()
        assert dim % heads == 0, "dim must be divisible by heads"
        self.heads    = heads
        self.head_dim = dim // heads

        self.norm_q  = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.W_q     = nn.Linear(dim, dim, bias=False)
        self.W_kv    = nn.Linear(dim, dim, bias=False)

        # attention vector a^T [h_i || h_j], one per head
        self.att   = nn.Parameter(torch.empty(heads, 2 * self.head_dim))
        self.leaky = nn.LeakyReLU(negative_slope=0.2)
        self.drop  = nn.Dropout(dropout)
        self.proj  = nn.Linear(dim, dim, bias=False)

        nn.init.xavier_uniform_(self.att.unsqueeze(0))
        nn.init.xavier_uniform_(self.W_q.weight)
        nn.init.xavier_uniform_(self.W_kv.weight)
        nn.init.xavier_uniform_(self.proj.weight)

    def forward(self, x_q, x_kv, nbr_mask):
        """
        x_q     : [B, dim]      -- query node features
        x_kv    : [B, K, dim]  -- neighbor features
        nbr_mask: [B, K] bool  -- True = valid (same or earlier layer)
        returns : [B, dim]     -- aggregated output  (add residual outside)
        """
        B, K, _ = x_kv.shape
        H, Dh   = self.heads, self.head_dim

        h_q  = self.W_q(self.norm_q(x_q))                       # [B, dim]
        h_kv = self.W_kv(self.norm_kv(x_kv))                    # [B, K, dim]

        h_q  = h_q.reshape(B, H, Dh)                            # [B, H, Dh]
        h_kv = h_kv.reshape(B, K, H, Dh)                        # [B, K, H, Dh]

        # e_ij = LeakyReLU(a^T [h_i || h_j])
        h_q_exp = h_q.unsqueeze(1).expand(-1, K, -1, -1)        # [B, K, H, Dh]
        cat     = torch.cat([h_q_exp, h_kv], dim=-1)             # [B, K, H, 2*Dh]
        e       = self.leaky(
            (cat * self.att.unsqueeze(0).unsqueeze(0)).sum(-1)   # [B, K, H]
        )

        # causal mask: future-layer neighbors -> -inf
        e = e.masked_fill(
            ~nbr_mask.unsqueeze(-1).to(e.device),
            torch.finfo(e.dtype).min,
        )

        alpha = e.softmax(dim=1)                                 # [B, K, H]
        alpha = torch.nan_to_num(alpha, nan=0.0)                # guard isolated nodes
        alpha = self.drop(alpha)

        out = (alpha.unsqueeze(-1) * h_kv).sum(dim=1)           # [B, H, Dh]
        return self.proj(out.reshape(B, H * Dh))                 # [B, dim]


class GAT(nn.Module):
    """
    Graph Attention Network (Velickovic et al. 2018) — pure baseline.

    Node features are derived solely from melt-pool images via CNN encoding.
    No spectral positional embeddings are used, keeping this consistent with
    the original GAT formulation.

    Forward inputs
    --------------
    x        : [B, C, H, W]      -- query image
    x_nbr    : [B, K, C, H, W]  -- K neighbor images
    nbr_mask : [B, K] bool       -- True if neighbor <= query layer

    Parameters
    ----------
    channels    : int   image channels (1 = grayscale)
    feat_dim    : int   CNN encoder output dim
    gat_dim     : int   hidden dim for GAT layers
    heads       : int   GAT attention heads (must divide gat_dim)
    depth       : int   number of GATConv + FFN blocks
    num_classes : int   (1 = regression)
    dropout     : float
    """
    def __init__(self, *, channels=1, feat_dim=256,
                 gat_dim=128, heads=4, depth=2,
                 num_classes=1, dropout=0.1):
        super().__init__()

        self.encoder = _CNNEncoder(channels=channels, feat_dim=feat_dim)

        # project feat_dim -> gat_dim  (no u concatenation)
        self.input_proj = nn.Sequential(
            nn.Linear(feat_dim, gat_dim),
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
        self.drop       = nn.Dropout(dropout)
        self.head       = nn.Linear(gat_dim, num_classes)
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def _encode(self, img):
        """
        CNN encode images only — no spectral embedding.
        img : [*lead, C, H, W]
        ->  : [*lead, gat_dim]
        """
        *lead, C, H, W = img.shape
        feat = self.encoder(img.reshape(-1, C, H, W))
        return self.input_proj(feat.reshape(*lead, -1))   # [*lead, gat_dim]

    def forward(self, x, x_nbr, nbr_mask):
        h_q  = self._encode(x)                            # [B, gat_dim]
        h_kv = self._encode(x_nbr)                        # [B, K, gat_dim]

        for gat_conv, ffn in self.blocks:
            h_q = h_q + gat_conv(h_q, h_kv, nbr_mask)
            h_q = h_q + ffn(h_q)

        return self.head(self.drop(self.final_norm(h_q)))  # [B, 1]

