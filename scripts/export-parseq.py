#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "torch",
#     "timm",
#     "pytorch-lightning",
#     "nltk",
#     "onnx",
#     "onnxscript",
#     "onnxruntime",
# ]
# ///
"""Export PARSeq (Permuted Autoregressive Sequence-to-Sequence) to ONNX with
cross-attention weight output for character-level positioning.

PARSeq is an English scene-text recognition model based on ViT encoder +
Transformer decoder. This script exports it with two outputs:
  - logits   [B, 26, 95]  — character predictions (94 printable ASCII + EOS)
  - attention [B, 26, 16] — horizontal attention weights (16 patch columns)

The attention output enables precise character-level positioning by mapping
each decoded character to a horizontal patch column in the input image.

Usage:
    uv run scripts/export-parseq.py [--output DIR]
"""
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn


def main():
    parser = argparse.ArgumentParser(description="Export PARSeq to ONNX with attention")
    parser.add_argument(
        "--output",
        default=".data/perception/ocr",
        help="Output directory (default: .data/perception/ocr)",
    )
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    onnx_path = os.path.join(args.output, "parseq_rec.onnx")
    charset_path = os.path.join(args.output, "parseq_charset.txt")

    print("=" * 60)
    print("PARSeq → ONNX Exporter (with cross-attention)")
    print("=" * 60)

    # Load pre-trained PARSeq
    print("\n[1/5] Loading PARSeq from torch.hub...")
    model = torch.hub.load("baudm/parseq", "parseq", pretrained=True, trust_repo=True)
    model.eval()
    inner = model.model  # unwrap LightningModule wrapper
    tok = model.tokenizer

    print(f"  Charset: {len(tok._itos)} classes (EOS + 94 ASCII + BOS + PAD)")
    print(f"  Image size: {model.hparams.img_size}")
    print(f"  Patch size: {model.hparams.patch_size}")
    print(f"  Max label length: {model.hparams.max_label_length}")

    # Save charset file (94 printable ASCII chars, one per line)
    # Index mapping: 0=EOS, 1..94=chars, 95=BOS, 96=PAD
    chars = list(tok._itos[1:95])  # skip EOS at 0, skip BOS/PAD at end
    with open(charset_path, "w") as f:
        for ch in chars:
            f.write(ch + "\n")
    print(f"  Charset saved: {charset_path} ({len(chars)} chars)")

    # Build ONNX wrapper
    print("\n[2/5] Building ONNX wrapper with attention output...")
    wrapper = PARSeqONNX(inner, bos_id=model.bos_id)
    wrapper.eval()

    # Validate wrapper output
    dummy = torch.randn(1, 3, 32, 128)
    with torch.no_grad():
        logits, attn = wrapper(dummy)
    print(f"  logits shape:    {logits.shape}")
    print(f"  attention shape: {attn.shape}")
    print(f"  attention sum:   {attn[0, 0].sum():.4f} (should be ~1.0)")

    # Export to ONNX
    print(f"\n[3/5] Exporting to {onnx_path}...")
    torch.onnx.export(
        wrapper,
        dummy,
        onnx_path,
        input_names=["images"],
        output_names=["logits", "attention"],
        dynamic_axes={
            "images": {0: "batch"},
            "logits": {0: "batch"},
            "attention": {0: "batch"},
        },
        opset_version=14,
        dynamo=False,
    )
    size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
    print(f"  File size: {size_mb:.1f} MB")

    # Validate with ONNX checker
    print("\n[4/5] Validating ONNX model...")
    import onnx

    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    for out in onnx_model.graph.output:
        dims = [d.dim_value or d.dim_param for d in out.type.tensor_type.shape.dim]
        print(f"  Output: {out.name} → {dims}")

    # Verify with ONNX Runtime
    print("\n[5/5] Verifying with ONNX Runtime...")
    import onnxruntime as ort

    sess = ort.InferenceSession(onnx_path)
    result = sess.run(None, {"images": dummy.numpy()})
    print(f"  ORT logits:    {result[0].shape}")
    print(f"  ORT attention: {result[1].shape}")
    print(f"  ORT attn sum:  {result[1][0, 0].sum():.4f}")

    # Compare PyTorch vs ORT
    with torch.no_grad():
        pt_logits, pt_attn = wrapper(dummy)
    logit_diff = np.abs(pt_logits.numpy() - result[0]).max()
    attn_diff = np.abs(pt_attn.numpy() - result[1]).max()
    print(f"  Max logit diff: {logit_diff:.6f}")
    print(f"  Max attn diff:  {attn_diff:.6f}")

    print(f"\n{'=' * 60}")
    print(f"✅ PARSeq exported successfully to {onnx_path}")
    print(f"   Charset: {charset_path}")
    print(f"   Model: 94 ASCII chars, 32×128 input, ImageNet normalization")
    print(f"{'=' * 60}")


class PARSeqONNX(nn.Module):
    """PARSeq wrapper that outputs logits + cross-attention weights.

    Uses non-autoregressive (NAR) single-pass decoding for ONNX compatibility.
    The cross-attention weights are extracted from the decoder's query→memory
    attention and reduced from 2D (8×16 patches) to 1D horizontal (16 columns).
    """

    def __init__(self, model: nn.Module, bos_id: int, max_len: int = 25):
        super().__init__()
        self.encoder = model.encoder
        self.text_embed = model.text_embed
        self.pos_queries = nn.Parameter(model.pos_queries.clone())
        self.head = model.head
        self.decoder_layers = model.decoder.layers
        self.decoder_norm = model.decoder.norm
        self.bos_id = bos_id
        self.max_len = max_len

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B = images.shape[0]
        memory = self.encoder(images)  # [B, 128, embed_dim]
        num_steps = self.max_len + 1  # 26

        # NAR decode: only BOS as input context
        bos = torch.full((B, 1), self.bos_id, dtype=torch.long, device=images.device)
        content = self.text_embed(bos)  # [B, 1, embed_dim]
        query = self.pos_queries[:, :num_steps].expand(B, -1, -1)  # [B, 26, embed_dim]

        all_ca: list[torch.Tensor] = []
        for i, layer in enumerate(self.decoder_layers):
            last = i == len(self.decoder_layers) - 1
            query_norm = layer.norm_q(query)
            content_norm = layer.norm_c(content)

            # Query stream: self-attn → cross-attn → FFN
            sa_out, _ = layer.self_attn(query_norm, content_norm, content_norm)
            query = query + layer.dropout1(sa_out)

            ca_out, ca_w = layer.cross_attn(layer.norm1(query), memory, memory)
            all_ca.append(ca_w)  # [B, 26, 128]
            query = query + layer.dropout2(ca_out)

            ff = layer.linear2(
                layer.dropout(layer.activation(layer.linear1(layer.norm2(query))))
            )
            query = query + layer.dropout3(ff)

            # Content stream (all layers except last)
            if not last:
                c_sa, _ = layer.self_attn(content_norm, content_norm, content_norm)
                content = content + layer.dropout1(c_sa)
                c_ca, _ = layer.cross_attn(layer.norm1(content), memory, memory)
                content = content + layer.dropout2(c_ca)
                c_ff = layer.linear2(
                    layer.dropout(layer.activation(layer.linear1(layer.norm2(content))))
                )
                content = content + layer.dropout3(c_ff)

        query = self.decoder_norm(query)
        logits = self.head(query)  # [B, 26, 95]

        # Average cross-attention across layers
        attn = torch.stack(all_ca).mean(dim=0)  # [B, 26, 128]

        # Reduce 2D patch attention (8 rows × 16 cols) to 1D horizontal
        # Image 32×128 with patches 4×8 → 8 rows, 16 columns
        attn_2d = attn.view(B, num_steps, 8, 16)
        attn_h = attn_2d.sum(dim=2)  # [B, 26, 16]

        return logits, attn_h


if __name__ == "__main__":
    main()
