#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "transformers>=4.30,<5",
#     "torch",
#     "timm",
#     "pillow",
#     "sentencepiece",
#     "onnx",
#     "onnxruntime",
# ]
# ///
"""Export TrOCR-Chinese (BEIT encoder + RoBERTa decoder) to ONNX with
cross-attention output for character-level positioning.

TrOCR is an encoder-decoder model exported as two ONNX files:
  - trocr_encoder.onnx  (~330MB) — ViT image encoder → [B, 577, 768]
  - trocr_decoder.onnx  (~850MB) — single-step decoder → (logits, attention)

The decoder outputs cross-attention weights reduced to 24 horizontal
columns (24×24 ViT patch grid → sum vertical → 24 columns), enabling
character-level positioning via centroid+FWHM.

Autoregressive decoding: the Rust backend runs the decoder in a loop,
feeding back the argmax token at each step, until EOS or max_length.

Default model: ZihCiLin/trocr-traditional-chinese-baseline (13176 vocab)
Override: --model <huggingface-model-id>

Usage:
    uv run scripts/export-trocr.py [--output DIR] [--model MODEL_ID]
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn


def main():
    parser = argparse.ArgumentParser(description="Export TrOCR-Chinese to ONNX")
    parser.add_argument(
        "--output",
        default=".data/perception/ocr",
        help="Output directory (default: .data/perception/ocr)",
    )
    parser.add_argument(
        "--model",
        default="ZihCiLin/trocr-traditional-chinese-baseline",
        help="HuggingFace model ID",
    )
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    enc_path = os.path.join(args.output, "trocr_encoder.onnx")
    dec_path = os.path.join(args.output, "trocr_decoder.onnx")
    vocab_path = os.path.join(args.output, "trocr_vocab.json")

    print("=" * 60)
    print("TrOCR-Chinese → ONNX Exporter (with cross-attention)")
    print("=" * 60)

    # Load model
    print(f"\n[1/6] Loading model: {args.model}")
    from transformers import VisionEncoderDecoderModel, AutoTokenizer

    model = VisionEncoderDecoderModel.from_pretrained(args.model)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    vocab_size = model.config.decoder.vocab_size
    enc_dim = model.encoder.config.hidden_size
    n_layers = model.config.decoder.decoder_layers

    print(f"  Encoder: ViT, hidden_size={enc_dim}")
    print(f"  Decoder: {n_layers} layers, vocab_size={vocab_size}")
    print(f"  Tokenizer: {len(tokenizer)} tokens")

    # Save vocab mapping (id → token)
    vocab = {}
    for token_id in range(len(tokenizer)):
        token = tokenizer.decode([token_id])
        vocab[str(token_id)] = token
    # Also save special token ids
    special = {
        "bos_id": tokenizer.bos_token_id or model.config.decoder_start_token_id or 2,
        "eos_id": tokenizer.eos_token_id or model.config.eos_token_id or 2,
        "pad_id": tokenizer.pad_token_id or 1,
        "vocab_size": vocab_size,
    }
    vocab["_special"] = special
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=0)
    print(f"  Vocab saved: {vocab_path}")
    print(f"  Special tokens: bos={special['bos_id']}, eos={special['eos_id']}, pad={special['pad_id']}")

    # Export encoder
    print(f"\n[2/6] Exporting encoder → {enc_path}")
    enc_wrapper = TrOCREncoderONNX(model.encoder)
    enc_wrapper.eval()

    dummy_img = torch.randn(1, 3, 384, 384)
    with torch.no_grad():
        enc_out = enc_wrapper(dummy_img)
    print(f"  Output shape: {enc_out.shape}")

    torch.onnx.export(
        enc_wrapper,
        dummy_img,
        enc_path,
        input_names=["pixel_values"],
        output_names=["encoder_hidden_states"],
        dynamic_axes={
            "pixel_values": {0: "batch"},
            "encoder_hidden_states": {0: "batch"},
        },
        opset_version=14,
        dynamo=False,
    )
    enc_mb = os.path.getsize(enc_path) / (1024 * 1024)
    print(f"  Size: {enc_mb:.1f} MB")

    # Export decoder
    print(f"\n[3/6] Exporting decoder → {dec_path}")
    dec_wrapper = TrOCRDecoderONNX(model.decoder)
    dec_wrapper.eval()

    dummy_enc = torch.randn(1, 577, enc_dim)
    dummy_ids = torch.tensor([[special["bos_id"]]])
    with torch.no_grad():
        logits, attn = dec_wrapper(dummy_ids, dummy_enc)
    print(f"  Logits: {logits.shape}, Attention: {attn.shape}")

    torch.onnx.export(
        dec_wrapper,
        (dummy_ids, dummy_enc),
        dec_path,
        input_names=["input_ids", "encoder_hidden_states"],
        output_names=["logits", "attention"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq_len"},
            "encoder_hidden_states": {0: "batch"},
            "logits": {0: "batch"},
            "attention": {0: "batch"},
        },
        opset_version=14,
        dynamo=False,
    )
    dec_mb = os.path.getsize(dec_path) / (1024 * 1024)
    print(f"  Size: {dec_mb:.1f} MB")

    # Verify encoder with ORT
    print(f"\n[4/6] Verifying encoder with ONNX Runtime...")
    import onnxruntime as ort

    enc_sess = ort.InferenceSession(enc_path)
    with torch.no_grad():
        pt_enc = enc_wrapper(dummy_img).numpy()
    ort_enc = enc_sess.run(None, {"pixel_values": dummy_img.numpy()})
    diff = np.abs(pt_enc - ort_enc[0]).max()
    print(f"  Max diff: {diff:.6f} ({'✅' if diff < 0.01 else '⚠️'})")

    # Verify decoder with ORT
    print(f"\n[5/6] Verifying decoder with ONNX Runtime...")
    dec_sess = ort.InferenceSession(dec_path)
    with torch.no_grad():
        pt_logits, pt_attn = dec_wrapper(dummy_ids, dummy_enc)
    ort_out = dec_sess.run(None, {
        "input_ids": dummy_ids.numpy(),
        "encoder_hidden_states": dummy_enc.numpy(),
    })
    logit_diff = np.abs(pt_logits.numpy() - ort_out[0]).max()
    attn_diff = np.abs(pt_attn.numpy() - ort_out[1]).max()
    print(f"  Logit diff: {logit_diff:.6f}, Attn diff: {attn_diff:.6f}")

    # End-to-end test with autoregressive decode
    print(f"\n[6/6] End-to-end autoregressive decode test...")
    enc_result = enc_sess.run(None, {"pixel_values": dummy_img.numpy()})
    enc_hidden = enc_result[0]

    tokens = [special["bos_id"]]
    all_attn = []
    for step in range(10):
        ids = np.array([tokens], dtype=np.int64)
        logits_np, attn_np = dec_sess.run(None, {
            "input_ids": ids,
            "encoder_hidden_states": enc_hidden,
        })
        next_id = int(logits_np[0, 0].argmax())
        if next_id == special["eos_id"]:
            break
        tokens.append(next_id)
        all_attn.append(attn_np[0, 0])

    decoded = tokenizer.decode(tokens[1:], skip_special_tokens=True)
    print(f"  Decoded ({len(tokens)-1} tokens): '{decoded}'")
    if all_attn:
        print(f"  Attention shape per step: [{len(all_attn)}, 24]")

    total_mb = enc_mb + dec_mb
    print(f"\n{'=' * 60}")
    print(f"✅ TrOCR-Chinese exported successfully!")
    print(f"   Encoder: {enc_path} ({enc_mb:.0f} MB)")
    print(f"   Decoder: {dec_path} ({dec_mb:.0f} MB)")
    print(f"   Vocab: {vocab_path} ({vocab_size} tokens)")
    print(f"   Total: {total_mb:.0f} MB")
    print(f"   Input: 384×384 RGB, ImageNet normalization")
    print(f"   Attention: 24 horizontal columns (24×24 patch grid)")
    print(f"{'=' * 60}")


class TrOCREncoderONNX(nn.Module):
    """ViT encoder wrapper for ONNX export."""

    def __init__(self, encoder: nn.Module):
        super().__init__()
        self.encoder = encoder

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.encoder(pixel_values, return_dict=False)[0]


class TrOCRDecoderONNX(nn.Module):
    """Single-step TrOCR decoder for ONNX export.

    Outputs logits for the LAST token position and averaged cross-attention
    reduced to 24 horizontal columns.
    """

    def __init__(self, decoder: nn.Module):
        super().__init__()
        self.decoder = decoder

    def forward(
        self,
        input_ids: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.decoder(
            input_ids=input_ids,
            encoder_hidden_states=encoder_hidden_states,
            output_attentions=True,
            return_dict=True,
        )
        # Only take logits for the last position
        logits = out.logits[:, -1:, :]  # [B, 1, vocab_size]

        # Average cross-attention across layers and heads for last query position
        attn_layers = []
        for ca in out.cross_attentions:
            # ca: [B, heads, seq_len, 577]
            attn_layers.append(ca[:, :, -1:, :])  # [B, heads, 1, 577]
        attn = torch.stack(attn_layers).mean(dim=0).mean(dim=1)  # [B, 1, 577]

        # Remove CLS token, reshape 2D patches → 1D horizontal
        attn_spatial = attn[:, :, 1:]  # [B, 1, 576]
        attn_2d = attn_spatial.view(-1, 1, 24, 24)
        attn_h = attn_2d.sum(dim=2)  # [B, 1, 24]

        return logits, attn_h


if __name__ == "__main__":
    main()
