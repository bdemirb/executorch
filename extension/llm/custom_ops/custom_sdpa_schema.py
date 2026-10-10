# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import threading

import torch

from torch.library import Library


class CustomSDPANativeOpUnavailable(RuntimeError):
    """Raised when the compiled custom SDPA op is unavailable."""


_CUSTOM_SDPA_SCHEMA_LOCK = threading.Lock()
_CUSTOM_SDPA_SCHEMA_REGISTERED = False
_CUSTOM_SDPA_SCHEMA_LIB = None
_CUSTOM_SDPA_FAKE_REGISTERED = False
_CUSTOM_SDPA_IMPLEMENTATION_LOCK = threading.Lock()
_CUSTOM_SDPA_IMPLEMENTATION_REGISTERED = False


def _validate_custom_sdpa_output_shape_inputs(query, key, value):
    assert query.dim() == key.dim() == value.dim() == 4, (
        "custom_sdpa requires 4D Q/K/V tensors but got "
        f"{query.dim()}D, {key.dim()}D, and {value.dim()}D"
    )
    supported_dtypes = (torch.float32, torch.bfloat16, torch.float16)
    assert query.dtype in supported_dtypes, (
        "custom_sdpa requires query to use float32, bfloat16, or float16 "
        f"but got {query.dtype}"
    )
    assert key.dtype == query.dtype == value.dtype, (
        "custom_sdpa requires matching Q/K/V dtypes but got "
        f"{query.dtype}, {key.dtype}, and {value.dtype}"
    )
    assert (
        tuple(query.shape[:-3]) == tuple(key.shape[:-3]) == tuple(value.shape[:-3])
    ), (
        "custom_sdpa does not support batch broadcasting; expected query, key, "
        f"and value to have matching batch shape but got {query.shape}, "
        f"{key.shape}, and {value.shape}"
    )
    assert query.shape[-1] == key.shape[-1] == value.shape[-1], (
        "custom_sdpa requires matching Q/K/V head dimensions but got "
        f"{query.shape[-1]}, {key.shape[-1]}, and {value.shape[-1]}"
    )
    assert key.shape[-2] == value.shape[-2], (
        "custom_sdpa requires key and value to have matching KV head counts "
        f"but got {key.shape[-2]} and {value.shape[-2]}"
    )
    assert key.shape[-2] > 0, "custom_sdpa requires at least one KV head"
    assert query.shape[-2] % key.shape[-2] == 0, (
        "custom_sdpa requires the query head count to match the KV head count "
        f"or be divisible by it, but got {query.shape[-2]} and {key.shape[-2]}"
    )


def custom_sdpa_output_shape(query, key, value):
    _validate_custom_sdpa_output_shape_inputs(query, key, value)
    return query.shape


def validate_custom_sdpa_args(
    query,
    key,
    value,
    start_pos,
    attn_mask=None,
    drpout_p=0.0,
    is_causal=False,
    scale=None,
):
    _ = (start_pos, drpout_p, scale)
    _validate_custom_sdpa_output_shape_inputs(query, key, value)
    assert not (
        attn_mask is not None and is_causal
    ), "attn_mask and is_causal cannot be set at the same time"
    if attn_mask is not None:
        assert attn_mask.dim() == 2, (
            "custom_sdpa requires a 2D attention mask but got " f"{attn_mask.dim()}D"
        )
        assert attn_mask.dtype == torch.float32, (
            "custom_sdpa requires attn_mask to use float32 but got "
            f"{attn_mask.dtype}"
        )
        assert attn_mask.shape == (query.shape[-3], key.shape[-3]), (
            "custom_sdpa requires attn_mask to match [q_len, kv_len], but got "
            f"{tuple(attn_mask.shape)} for q_len={query.shape[-3]} "
            f"and kv_len={key.shape[-3]}"
        )


def _has_custom_sdpa_meta_impl() -> bool:
    try:
        return torch._C._dispatch_has_kernel_for_dispatch_key(
            "llama::custom_sdpa",
            "Meta",
        )
    except RuntimeError:
        return False


def custom_sdpa_fake(
    query,
    key,
    value,
    start_pos,
    attn_mask=None,
    drpout_p=0.0,
    is_causal=False,
    scale=None,
):
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
    return torch.empty(
        custom_sdpa_output_shape(query, key, value),
        dtype=query.dtype,
        device=query.device,
    )


def register_custom_sdpa_fake() -> bool:
    global _CUSTOM_SDPA_FAKE_REGISTERED

    if _CUSTOM_SDPA_FAKE_REGISTERED:
        return False
    if _has_custom_sdpa_meta_impl():
        _CUSTOM_SDPA_FAKE_REGISTERED = True
        return False

    torch.library.register_fake("llama::custom_sdpa")(custom_sdpa_fake)
    _CUSTOM_SDPA_FAKE_REGISTERED = True
    return True


def ensure_custom_sdpa_schema() -> None:
    global _CUSTOM_SDPA_SCHEMA_LIB
    global _CUSTOM_SDPA_SCHEMA_REGISTERED

    if _CUSTOM_SDPA_SCHEMA_REGISTERED:
        return

    try:
        torch.ops.llama.custom_sdpa.default
    except (AttributeError, RuntimeError):
        pass
    else:
        _CUSTOM_SDPA_SCHEMA_REGISTERED = True
        register_custom_sdpa_fake()
        return

    with _CUSTOM_SDPA_SCHEMA_LOCK:
        if _CUSTOM_SDPA_SCHEMA_REGISTERED:
            return

        try:
            torch.ops.llama.custom_sdpa.default
        except (AttributeError, RuntimeError):
            _CUSTOM_SDPA_SCHEMA_LIB = Library("llama", "FRAGMENT")
            _CUSTOM_SDPA_SCHEMA_LIB.define(
                "custom_sdpa(Tensor query, Tensor key, Tensor value, SymInt start_pos, "
                "Tensor? attn_mask=None, float drpout_p=0.0, bool is_causal=False, "
                "float? scale=None) -> Tensor"
            )

        _CUSTOM_SDPA_SCHEMA_REGISTERED = True

    register_custom_sdpa_fake()


def ensure_custom_sdpa_registered() -> None:
    global _CUSTOM_SDPA_IMPLEMENTATION_REGISTERED
    global _CUSTOM_SDPA_SCHEMA_REGISTERED

    if _CUSTOM_SDPA_IMPLEMENTATION_REGISTERED:
        return

    with _CUSTOM_SDPA_IMPLEMENTATION_LOCK:
        if _CUSTOM_SDPA_IMPLEMENTATION_REGISTERED:
            return

        try:
            from executorch.extension.llm.custom_ops import custom_ops  # noqa: F401
        except CustomSDPANativeOpUnavailable as exc:
            ensure_custom_sdpa_schema()
            from executorch.extension.llm.custom_ops import custom_sdpa_fallback

            logging.warning(
                "Falling back to Python llama.custom_sdpa registration: %s", exc
            )
            custom_sdpa_fallback.register_custom_sdpa_fallback_impl()
        else:
            _CUSTOM_SDPA_SCHEMA_REGISTERED = True

        _CUSTOM_SDPA_IMPLEMENTATION_REGISTERED = True
