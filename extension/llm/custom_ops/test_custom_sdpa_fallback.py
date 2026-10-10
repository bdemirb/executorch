# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import builtins
import importlib
import types
import unittest
from unittest import mock

import torch
import torch.nn.functional as F

from executorch.extension.llm.custom_ops import custom_sdpa_schema as schema


class CustomSDPAFallbackTest(unittest.TestCase):
    def test_fallback_respects_start_pos_for_causal_decode(self):
        from executorch.extension.llm.custom_ops import custom_sdpa_fallback

        torch.manual_seed(0)
        query = torch.randn(1, 2, 1, 4)
        key = torch.randn(1, 5, 1, 4)
        value = torch.randn(1, 5, 1, 4)
        start_pos = 3

        actual = custom_sdpa_fallback.custom_sdpa_fallback_impl(
            query,
            key,
            value,
            start_pos,
            attn_mask=None,
            drpout_p=0.0,
            is_causal=True,
            scale=None,
        )

        query_t = query.transpose(1, 2)
        key_t = key[:, : start_pos + query.size(1)].transpose(1, 2)
        value_t = value[:, : start_pos + query.size(1)].transpose(1, 2)
        attn_mask = custom_sdpa_fallback._build_offset_causal_mask(
            query_t, key_t, start_pos
        )
        expected = F.scaled_dot_product_attention(
            query_t,
            key_t,
            value_t,
            attn_mask=attn_mask,
            dropout_p=0.0,
            is_causal=False,
            scale=None,
        ).transpose(1, 2)

        self.assertTrue(torch.allclose(actual, expected))

    def test_fallback_rejects_causal_prefix_beyond_available_kv_len(self):
        from executorch.extension.llm.custom_ops import custom_sdpa_fallback

        torch.manual_seed(0)
        query = torch.randn(1, 4, 1, 4)
        key = torch.randn(1, 2, 1, 4)
        value = torch.randn(1, 2, 1, 4)

        with self.assertRaisesRegex(AssertionError, "must fit within the key cache"):
            custom_sdpa_fallback.custom_sdpa_fallback_impl(
                query,
                key,
                value,
                start_pos=0,
                attn_mask=None,
                drpout_p=0.0,
                is_causal=True,
                scale=None,
            )

    def test_fallback_uses_kv_prefix_for_non_causal_unmasked_sdpa(self):
        from executorch.extension.llm.custom_ops import custom_sdpa_fallback

        torch.manual_seed(0)
        query = torch.randn(1, 2, 4, 4)
        key = torch.randn(1, 5, 2, 4)
        value = torch.randn(1, 5, 2, 4)

        actual = custom_sdpa_fallback.custom_sdpa_fallback_impl(
            query,
            key,
            value,
            start_pos=0,
            attn_mask=None,
            drpout_p=0.0,
            is_causal=False,
            scale=None,
        )

        query_t = query.transpose(1, 2)
        key_t = key[:, : query.size(1)].transpose(1, 2).repeat_interleave(2, dim=1)
        value_t = value[:, : query.size(1)].transpose(1, 2).repeat_interleave(2, dim=1)
        expected = F.scaled_dot_product_attention(
            query_t,
            key_t,
            value_t,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=False,
            scale=None,
        ).transpose(1, 2)

        self.assertTrue(torch.allclose(actual, expected))


class CustomSDPASchemaRegistrationTest(unittest.TestCase):
    def setUp(self):
        self.prev_fake_registered = schema._CUSTOM_SDPA_FAKE_REGISTERED
        self.prev_impl_registered = schema._CUSTOM_SDPA_IMPLEMENTATION_REGISTERED
        self.prev_schema_registered = schema._CUSTOM_SDPA_SCHEMA_REGISTERED

    def tearDown(self):
        schema._CUSTOM_SDPA_FAKE_REGISTERED = self.prev_fake_registered
        schema._CUSTOM_SDPA_IMPLEMENTATION_REGISTERED = self.prev_impl_registered
        schema._CUSTOM_SDPA_SCHEMA_REGISTERED = self.prev_schema_registered

    def test_prefers_compiled_import_before_defining_python_schema(self):
        schema._CUSTOM_SDPA_IMPLEMENTATION_REGISTERED = False
        schema._CUSTOM_SDPA_SCHEMA_REGISTERED = False
        calls = []
        real_import = builtins.__import__

        def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "executorch.extension.llm.custom_ops":
                calls.append(schema._CUSTOM_SDPA_SCHEMA_REGISTERED)
                return types.SimpleNamespace(custom_ops=object())
            return real_import(name, globals, locals, fromlist, level)

        with mock.patch("builtins.__import__", side_effect=fake_import):
            with mock.patch.object(
                schema,
                "ensure_custom_sdpa_schema",
                side_effect=AssertionError(
                    "Python schema should not be defined before compiled import."
                ),
            ):
                schema.ensure_custom_sdpa_registered()

        self.assertEqual(calls, [False])
        self.assertTrue(schema._CUSTOM_SDPA_IMPLEMENTATION_REGISTERED)
        self.assertTrue(schema._CUSTOM_SDPA_SCHEMA_REGISTERED)

    def test_fallback_impl_is_not_registered_when_compiled_impl_exists(self):
        from executorch.extension.llm.custom_ops import custom_sdpa_fallback

        with mock.patch(
            "executorch.extension.llm.custom_ops.custom_sdpa_fallback."
            "_has_custom_sdpa_composite_impl",
            return_value=True,
        ):
            with mock.patch.object(custom_sdpa_fallback, "impl") as impl_mock:
                self.assertFalse(
                    custom_sdpa_fallback.register_custom_sdpa_fallback_impl()
                )
                impl_mock.assert_not_called()

    def test_shared_fake_is_not_registered_when_meta_impl_exists(self):
        schema._CUSTOM_SDPA_FAKE_REGISTERED = False
        with mock.patch(
            "executorch.extension.llm.custom_ops.custom_sdpa_schema."
            "_has_custom_sdpa_meta_impl",
            return_value=True,
        ):
            with mock.patch("torch.library.register_fake") as register_fake_mock:
                self.assertFalse(schema.register_custom_sdpa_fake())
                register_fake_mock.assert_not_called()

    def test_fallback_registration_is_explicit(self):
        fallback = importlib.reload(
            importlib.import_module(
                "executorch.extension.llm.custom_ops.custom_sdpa_fallback"
            )
        )
        with mock.patch.object(fallback, "impl") as impl_mock:
            with mock.patch.object(
                fallback,
                "_has_custom_sdpa_composite_impl",
                return_value=False,
            ):
                fallback.register_custom_sdpa_fallback_impl()
        impl_mock.assert_called_once()

    def test_native_load_errors_are_not_silently_swallowed(self):
        schema._CUSTOM_SDPA_IMPLEMENTATION_REGISTERED = False
        real_import = builtins.__import__

        def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "executorch.extension.llm.custom_ops" and fromlist == (
                "custom_ops",
            ):
                raise OSError("broken custom ops library")
            return real_import(name, globals, locals, fromlist, level)

        with mock.patch("builtins.__import__", side_effect=fake_import):
            with self.assertRaisesRegex(OSError, "broken custom ops library"):
                schema.ensure_custom_sdpa_registered()

    def test_native_unavailable_uses_python_fallback(self):
        schema._CUSTOM_SDPA_IMPLEMENTATION_REGISTERED = False
        fallback = importlib.import_module(
            "executorch.extension.llm.custom_ops.custom_sdpa_fallback"
        )
        real_import = builtins.__import__

        def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "executorch.extension.llm.custom_ops" and fromlist == (
                "custom_ops",
            ):
                raise schema.CustomSDPANativeOpUnavailable("native custom ops missing")
            return real_import(name, globals, locals, fromlist, level)

        with mock.patch("builtins.__import__", side_effect=fake_import):
            with mock.patch.object(
                fallback,
                "register_custom_sdpa_fallback_impl",
                wraps=fallback.register_custom_sdpa_fallback_impl,
            ) as register_mock:
                schema.ensure_custom_sdpa_registered()
        register_mock.assert_called_once()

    def test_fake_rejects_non_float32_mask(self):
        q = torch.randn(1, 4, 3, 8)
        k = torch.randn(1, 5, 1, 8)
        v = torch.randn(1, 5, 1, 8)
        mask = torch.ones(4, 5, dtype=torch.float16)

        with self.assertRaisesRegex(AssertionError, "float32"):
            schema.custom_sdpa_fake(q, k, v, 0, mask, 0.0, False, None)

    def test_fallback_ignores_dropout_like_native_kernel(self):
        from executorch.extension.llm.custom_ops import custom_sdpa_fallback

        torch.manual_seed(0)
        query = torch.randn(1, 2, 4, 4)
        key = torch.randn(1, 5, 2, 4)
        value = torch.randn(1, 5, 2, 4)

        actual = custom_sdpa_fallback.custom_sdpa_fallback_impl(
            query,
            key,
            value,
            start_pos=0,
            attn_mask=None,
            drpout_p=0.7,
            is_causal=False,
            scale=None,
        )
        expected = custom_sdpa_fallback.custom_sdpa_fallback_impl(
            query,
            key,
            value,
            start_pos=0,
            attn_mask=None,
            drpout_p=0.0,
            is_causal=False,
            scale=None,
        )
        self.assertTrue(torch.allclose(actual, expected))


if __name__ == "__main__":
    unittest.main()
