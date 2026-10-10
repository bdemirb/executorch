# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
# pyre-ignore-all-errors

import torch

from executorch.extension.llm.custom_ops.custom_sdpa_schema import (
    ensure_custom_sdpa_schema,
    validate_custom_sdpa_args,
)
from torch.library import impl, Library

custom_sdpa_fallback_lib = Library("llama", "IMPL")


def _has_custom_sdpa_composite_impl() -> bool:
    try:
        return torch._C._dispatch_has_kernel_for_dispatch_key(
            "llama::custom_sdpa",
            "CompositeExplicitAutograd",
        )
    except RuntimeError:
        return False


def _slice_kv_prefix_for_decode(
    key: torch.Tensor,
    value: torch.Tensor,
    start_pos: int,
    query_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    usable_kv_len = start_pos + query_len
    assert usable_kv_len <= key.size(2), (
        "start_pos + query_len must fit within the key cache, but got "
        f"{usable_kv_len} > {key.size(2)}"
    )
    assert usable_kv_len <= value.size(2), (
        "start_pos + query_len must fit within the value cache, but got "
        f"{usable_kv_len} > {value.size(2)}"
    )
    return (
        key.narrow(2, 0, usable_kv_len),
        value.narrow(2, 0, usable_kv_len),
    )


def _build_offset_causal_mask(
    query: torch.Tensor,
    key: torch.Tensor,
    start_pos: int,
) -> torch.Tensor:
    query_len = query.size(2)
    key_len = key.size(2)
    query_positions = torch.arange(query_len, device=query.device).unsqueeze(1)
    key_positions = torch.arange(key_len, device=key.device).unsqueeze(0)
    return key_positions <= (start_pos + query_positions)


def custom_sdpa_fallback_impl(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    start_pos: int,
    attn_mask: torch.Tensor | None = None,
    drpout_p: float = 0.0,
    is_causal: bool = False,
    scale: float | None = None,
) -> torch.Tensor:
    validate_custom_sdpa_args(
        query,
        key,
        value,
        start_pos,
        attn_mask,
        drpout_p,
        is_causal,
        scale,
    )

    query = query.transpose(1, 2)
    key = key.transpose(1, 2)
    value = value.transpose(1, 2)
    query_len = query.size(2)

    num_heads_q = query.size(1)
    num_heads_kv = key.size(1)
    if num_heads_q != num_heads_kv:
        assert num_heads_q % num_heads_kv == 0
        repeats = num_heads_q // num_heads_kv
        key = key.repeat_interleave(repeats, dim=1)
        value = value.repeat_interleave(repeats, dim=1)

    if attn_mask is None:
        key, value = _slice_kv_prefix_for_decode(key, value, start_pos, query_len)
        if is_causal:
            attn_mask = _build_offset_causal_mask(query, key, start_pos)
            is_causal = False

    output = torch.nn.functional.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=attn_mask,
        dropout_p=0.0,
        is_causal=is_causal,
        scale=scale,
    )
    return output.transpose(1, 2).contiguous()


def register_custom_sdpa_fallback_impl() -> bool:
    ensure_custom_sdpa_schema()
    if _has_custom_sdpa_composite_impl():
        return False
    impl(custom_sdpa_fallback_lib, "custom_sdpa", "CompositeExplicitAutograd")(
        custom_sdpa_fallback_impl
    )
    return True
