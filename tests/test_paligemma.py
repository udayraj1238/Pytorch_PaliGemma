"""Shape and KV-cache correctness checks for the from-scratch PaliGemma.

Run:  python -m pytest -q
Uses a 2-layer config so it runs on CPU in seconds; the vision tower test
uses the full default SigLIP config.
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modeling_siglip import SiglipVisionConfig, SiglipVisionModel  # noqa: E402
from modelling_gemma import (  # noqa: E402
    KVCache,
    PaliGemmaConfig,
    PaliGemmaForConditionalGeneration,
)

IMAGE_TOKEN = 256


def small_model():
    cfg = PaliGemmaConfig(
        vision_config=dict(num_hidden_layers=2),
        image_token_index=IMAGE_TOKEN,
        text_config=dict(vocab_size=300, hidden_size=2048, intermediate_size=512,
                         num_hidden_layers=2, num_attention_heads=8,
                         num_key_value_heads=1, head_dim=256),
    )
    return cfg, PaliGemmaForConditionalGeneration(cfg).eval()


def test_siglip_default_config_gives_196_tokens():
    cfg = SiglipVisionConfig()
    assert (cfg.num_hidden_layers, cfg.hidden_size, cfg.image_size, cfg.patch_size) == (12, 768, 224, 16)
    with torch.no_grad():
        out = SiglipVisionModel(cfg).eval()(torch.randn(1, 3, 224, 224))
    assert out.shape == (1, 196, 768)


def test_projector_maps_768_to_2048():
    _, model = small_model()
    n = sum(p.numel() for p in model.multi_modal_projector.parameters())
    assert n == 768 * 2048 + 2048  # 1,574,912 parameters
    with torch.no_grad():
        out = model.multi_modal_projector(torch.randn(1, 196, 768))
    assert out.shape == (1, 196, 2048)


def test_kv_cached_decoding_matches_full_forward():
    torch.manual_seed(0)
    _, model = small_model()
    pixels = torch.randn(1, 3, 224, 224)
    prompt = torch.cat([torch.full((1, 196), IMAGE_TOKEN), torch.tensor([[5, 6, 7, 8]])], dim=1)
    next_tok = torch.tensor([[9]])
    full_ids = torch.cat([prompt, next_tok], dim=1)

    with torch.no_grad():
        full = model(input_ids=full_ids, pixel_values=pixels,
                     attention_mask=torch.ones_like(full_ids), kv_cache=KVCache())["logits"][:, -1]
        cache = KVCache()
        model(input_ids=prompt, pixel_values=pixels,
              attention_mask=torch.ones_like(prompt), kv_cache=cache)
        step = model(input_ids=next_tok, pixel_values=pixels,
                     attention_mask=torch.ones_like(full_ids), kv_cache=cache)["logits"][:, -1]

    assert torch.allclose(full, step, atol=1e-4)
