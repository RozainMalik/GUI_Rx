import os

os.environ["QWEN35_TORCH_KERNELS"] = "1"  # tests run on CPU tensors: keep fla's Triton kernels out (also for subprocesses)
import gui_aima.qwen35  # noqa: E402,F401  (applies the switch before transformers' qwen3_5 modeling is imported)
import pytest  # noqa: E402
import torch  # noqa: E402
from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration  # noqa: E402

IMAGE_TOKEN, VIDEO_TOKEN, VSTART, VEND = 300, 301, 302, 303


def tiny_config(vocab_size: int = 320) -> Qwen3_5Config:
    """8 layers (full attention at 3 and 7), 4 heads / 2 KV heads, head_dim 32, 32-px visual tokens."""
    layer_types = ["linear_attention"] * 3 + ["full_attention"]
    return Qwen3_5Config(
        text_config=dict(
            vocab_size=vocab_size, hidden_size=64, intermediate_size=128, num_hidden_layers=8,
            layer_types=layer_types * 2, num_attention_heads=4, num_key_value_heads=2, head_dim=32,
            linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=16,
            linear_value_head_dim=16, linear_conv_kernel_dim=4, max_position_embeddings=4096,
            rope_parameters=dict(rope_type="default", rope_theta=10000.0, partial_rotary_factor=0.25,
                                 mrope_section=[2, 1, 1], mrope_interleaved=True),
            tie_word_embeddings=False,
        ),
        vision_config=dict(depth=1, hidden_size=32, intermediate_size=64, num_heads=2, out_hidden_size=64,
                           patch_size=16, spatial_merge_size=2, temporal_patch_size=2,
                           num_position_embeddings=64, deepstack_visual_indexes=[]),
        image_token_id=IMAGE_TOKEN, video_token_id=VIDEO_TOKEN,
        vision_start_token_id=VSTART, vision_end_token_id=VEND, tie_word_embeddings=False,
    )


def make_mm_inputs(cfg, batch_lengths, grid=(1, 4, 4), seed=0):
    """Right-padded batch: [text x2, vision_start, image x (h*w/4), vision_end, text ...] per sample."""
    g = torch.Generator().manual_seed(seed)
    t, h, w = grid
    n_img = t * h * w // 4
    patch_dim = 3 * 2 * 16 * 16
    max_len = max(batch_lengths)
    ids = torch.zeros(len(batch_lengths), max_len, dtype=torch.long)
    mask = torch.zeros_like(ids)
    for b, n in enumerate(batch_lengths):
        assert n >= n_img + 6
        seq = torch.randint(0, 250, (n,), generator=g)
        seq[2] = VSTART
        seq[3:3 + n_img] = IMAGE_TOKEN
        seq[3 + n_img] = VEND
        ids[b, :n] = seq
        mask[b, :n] = 1
    return dict(
        input_ids=ids,
        attention_mask=mask,
        pixel_values=torch.randn(len(batch_lengths) * t * h * w, patch_dim, generator=g),
        image_grid_thw=torch.tensor([list(grid)] * len(batch_lengths)),
        mm_token_type_ids=(ids == IMAGE_TOKEN).int(),
    )


@pytest.fixture
def tiny_model():
    torch.manual_seed(0)
    model = Qwen3_5ForConditionalGeneration(tiny_config())
    model.config._attn_implementation = "eager"
    model.config.text_config._attn_implementation = "eager"
    return model.float().eval()
