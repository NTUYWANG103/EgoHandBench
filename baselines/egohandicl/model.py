"""EgoHandICL network from the released ``handiclmodel_text.py``.

The released repository omits ``handicl.egohandicl_new``, which its inference
script imports. This file keeps the released ``HandICL`` layers and forward pass
unchanged except for these changes:
- ``mano_dim`` uses the inference script's 122-value layout.
- Qwen-7B hidden states arrive as inputs, so the text model is not loaded at import.
- An optional key-padding mask makes batched training equal to unpadded single-text calls.
- The ViT has qkv biases, so every parameter of the generic ImageNet ViT-B/16 initialization
  loads. It keeps the released HaMeR ViT forward (2-pixel patch padding; the class-token
  position embedding is added to every patch), so its frozen features are not identical to timm's.
- ``cross_residual`` (default False, as released) adds the query back after each cross-attention.
  The released cross-attention returns only attended image values, so the MANO tokens steer
  attention but their content never reaches the predictor.
- ``residual_output`` (default False, as released) predicts each target as a correction of
  its pair's coarse input (m1a for the template, m2a for the query).
"""

import torch
import torch.nn as nn

from .vit import ViT

# Per hand: global-orient axis-angle (3), hand-pose axis-angle (45), betas (10), cam (3);
# the vector concatenates [left, right] (egohandicl_infer_new.process_mano).
HAND_DIM = 61
MANO_DIM = 2 * HAND_DIM


class TextCompressor(nn.Module):
    def __init__(self, input_dim, output_dim, num_slots=4):
        super().__init__()
        self.num_slots = num_slots
        self.attn = nn.MultiheadAttention(embed_dim=input_dim, num_heads=8, batch_first=True)
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, text_feat, padding_mask=None):
        B, T, D = text_feat.shape
        queries = torch.zeros(B, self.num_slots, D, device=text_feat.device, dtype=text_feat.dtype)
        queries = queries.to(torch.float32)
        text_feat = text_feat.to(torch.float32)
        attended, _ = self.attn(queries, text_feat, text_feat, key_padding_mask=padding_mask)
        return self.linear(attended)


class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, dropout=0.1, residual=False):
        super().__init__()
        self.residual = residual
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, context):
        B, N, C = x.shape
        q = self.q_proj(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k_proj(context).reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v_proj(context).reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        out = self.out_proj((attn @ v).transpose(1, 2).reshape(B, N, C))
        return x + out if self.residual else out


class HandICL(nn.Module):
    def __init__(self, hidden_dim=128, num_heads=4, text_dim=4096, cross_residual=False, residual_output=False):
        super().__init__()
        self.residual_output = residual_output
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.mano_dim = MANO_DIM
        self.num_pairs = 4
        self.vit = ViT(
            img_size=224, patch_size=16, in_chans=3, embed_dim=768, depth=12, num_heads=12,
            mlp_ratio=4, qkv_bias=True, drop_rate=0.0, drop_path_rate=0.0,
        )
        self.img_proj = nn.Linear(768, self.hidden_dim)
        self.mano_encoder = nn.Sequential(
            nn.Linear(self.mano_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.cross_attn1 = CrossAttention(self.hidden_dim, self.num_heads, residual=cross_residual)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.hidden_dim, nhead=self.num_heads,
            dim_feedforward=4 * self.hidden_dim, dropout=0.1,
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=4)
        self.text_compressor = TextCompressor(input_dim=text_dim, output_dim=self.hidden_dim, num_slots=4)
        self.fusion = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(self.hidden_dim),
        )
        self.cross_attn2 = CrossAttention(self.hidden_dim, self.num_heads, residual=cross_residual)
        self.predictor = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.mano_dim),
        )

    def forward(self, batch):
        """Return the four predicted 122-value vectors [m1a, m1b, m2a, m2b], concatenated."""
        B = batch["img1"].shape[0]
        with torch.no_grad():
            img1_feat = self.vit(batch["img1"])  # [B, 768, 14, 14]
            img2_feat = self.vit(batch["img2"])
        img1_feat = img1_feat.permute(0, 2, 3, 1).reshape(B, 196, 768)
        img2_feat = img2_feat.permute(0, 2, 3, 1).reshape(B, 196, 768)
        img1_feat = self.img_proj(img1_feat)
        img2_feat = self.img_proj(img2_feat)

        compressed_text1 = self.text_compressor(batch["text1"], batch.get("text1_mask"))
        compressed_text2 = self.text_compressor(batch["text2"], batch.get("text2_mask"))

        encoded_params = []
        for key in ("m1a", "m1b", "m2a", "m2b"):
            param = batch["mano_params"][key].reshape(B, -1)
            encoded_params.append(self.mano_encoder(param).unsqueeze(1))
        mano_feat = torch.cat(encoded_params, dim=1)  # [B, 4, hidden]

        attn_feat = self.cross_attn1(mano_feat, img1_feat)
        fusion_feat = self.fusion(torch.cat([attn_feat, compressed_text1], dim=-1))
        fusion_feat = fusion_feat.transpose(0, 1)
        img1_feat = img1_feat.transpose(0, 1)
        decoded_feat = self.transformer_decoder(fusion_feat, img1_feat).transpose(0, 1)

        final_feat = self.cross_attn2(decoded_feat, img2_feat)
        final_feat = self.fusion(torch.cat([final_feat, compressed_text2], dim=-1))

        predictions = [self.predictor(final_feat[:, i]) for i in range(4)]
        if self.residual_output:
            # Each target is predicted as a correction of its pair's coarse input.
            base = [batch["mano_params"][k].reshape(B, -1) for k in ("m1a", "m1a", "m2a", "m2a")]
            predictions = [p.float() + b.float() for p, b in zip(predictions, base)]
        return torch.cat(predictions, dim=1)
