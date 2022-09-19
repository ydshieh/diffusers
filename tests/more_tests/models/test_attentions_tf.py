import numpy as np
import torch
import tensorflow as tf
import unittest

from diffusers.models.attention_tf import TFGEGLU, TFFeedForward, TFAttentionBlock, TFCrossAttention, TFBasicTransformerBlock, TFSpatialTransformer
from diffusers.models.attention import GEGLU, FeedForward, AttentionBlock, CrossAttention, BasicTransformerBlock, SpatialTransformer

from transformers import load_pytorch_model_in_tf2_model


class TFGEGLUTest(unittest.TestCase):

    def test_pt_tf_default(self):
        N, C = (1, 16)
        C_out = 32

        sample = np.random.default_rng().standard_normal(size=(N, C), dtype=np.float32)

        pt_sample = torch.tensor(sample)
        tf_sample = tf.constant(sample)

        pt_layer = GEGLU(dim_in=C, dim_out=C_out)
        tf_layer = TFGEGLU(dim_in=C, dim_out=C_out)

        # init. TF weights
        _ = tf_layer(tf_sample)
        # Load PT weights
        tf_layer.base_model_prefix = ""
        tf_layer._keys_to_ignore_on_load_missing = []
        tf_layer._keys_to_ignore_on_load_unexpected = []
        load_pytorch_model_in_tf2_model(tf_layer, pt_layer, tf_inputs=tf_sample, allow_missing_keys=False)

        with torch.no_grad():
            pt_output = pt_layer(pt_sample)
        tf_output = tf_layer(tf_sample)

        max_diff = np.amax(np.abs(pt_output.numpy() - tf_output.numpy()))
        assert max_diff < 1e-6


class TFFeedForwardTest(unittest.TestCase):

    def test_pt_tf_default(self):
        N, C = (1, 16)
        C_out = 32

        sample = np.random.default_rng().standard_normal(size=(N, C), dtype=np.float32)

        pt_sample = torch.tensor(sample)
        tf_sample = tf.constant(sample)

        pt_layer = FeedForward(dim=C, dim_out=C_out)
        tf_layer = TFFeedForward(dim=C, dim_out=C_out)

        # init. TF weights
        _ = tf_layer(tf_sample)
        # Load PT weights
        tf_layer.base_model_prefix = ""
        tf_layer._keys_to_ignore_on_load_missing = []
        tf_layer._keys_to_ignore_on_load_unexpected = []
        load_pytorch_model_in_tf2_model(tf_layer, pt_layer, tf_inputs=tf_sample, allow_missing_keys=False)

        with torch.no_grad():
            pt_output = pt_layer(pt_sample)
        tf_output = tf_layer(tf_sample)

        max_diff = np.amax(np.abs(pt_output.numpy() - tf_output.numpy()))
        assert max_diff < 1e-6


class TFAttentionBlockTest(unittest.TestCase):

    def test_pt_tf_default(self):
        N, H, W, C = (1, 16, 16, 6)

        sample = np.random.default_rng().standard_normal(size=(N, C, H, W), dtype=np.float32)

        pt_sample = torch.tensor(sample)
        # (N, C, H, W) -> (N, H, W, C) for TF
        tf_sample = tf.transpose(tf.constant(sample), perm=(0, 2, 3, 1))

        pt_layer = AttentionBlock(channels=C, num_head_channels=2, num_groups=3)
        tf_layer = TFAttentionBlock(channels=C, num_head_channels=2, num_groups=3)

        # init. TF weights
        _ = tf_layer(tf_sample)
        # Load PT weights
        tf_layer.base_model_prefix = ""
        tf_layer._keys_to_ignore_on_load_missing = []
        tf_layer._keys_to_ignore_on_load_unexpected = []
        load_pytorch_model_in_tf2_model(tf_layer, pt_layer, tf_inputs=tf_sample, allow_missing_keys=False)

        with torch.no_grad():
            pt_output = pt_layer(pt_sample)
        tf_output = tf_layer(tf_sample)
        # (N, H, W, C) -> (N, C, H, W)
        tf_output = tf.transpose(tf_output, perm=(0, 3, 1, 2))

        max_diff = np.amax(np.abs(pt_output.numpy() - tf_output.numpy()))
        assert max_diff < 1e-6


class TFCrossAttentionTest(unittest.TestCase):

    def test_pt_tf_default(self):
        N, T_query, T_context, query_dim, context_dim, heads, dim_head = (1, 5, 3, 8, 4, 2, 8)

        sample = np.random.default_rng().standard_normal(size=(N, T_query, query_dim), dtype=np.float32)
        context = np.random.default_rng().standard_normal(size=(N, T_context, context_dim), dtype=np.float32)

        pt_sample = torch.tensor(sample, dtype=torch.float32)
        tf_sample = tf.constant(sample)

        pt_context = torch.tensor(context, dtype=torch.float32)
        tf_context = tf.constant(context)

        pt_layer = CrossAttention(query_dim=query_dim, context_dim=context_dim, heads=heads, dim_head=dim_head)
        tf_layer = TFCrossAttention(query_dim=query_dim, context_dim=context_dim, heads=heads, dim_head=dim_head)

        # init. TF weights
        _ = tf_layer(tf_sample, context=tf_context)
        # Load PT weights
        tf_layer.base_model_prefix = ""
        tf_layer._keys_to_ignore_on_load_missing = []
        tf_layer._keys_to_ignore_on_load_unexpected = []
        # TODO: Clean up
        load_pytorch_model_in_tf2_model(tf_layer, pt_layer, tf_inputs={"hidden_states": tf_sample, "context": tf_context}, allow_missing_keys=False)

        with torch.no_grad():
            pt_output = pt_layer(pt_sample, context=pt_context)
        tf_output = tf_layer(tf_sample, context=tf_context)

        max_diff = np.amax(np.abs(pt_output.numpy() - tf_output.numpy()))
        assert max_diff < 1e-6

    def test_pt_tf_default_without_context(self):
        N, T_query, T_context, query_dim, context_dim, heads, dim_head = (1, 5, 3, 8, None, 2, 8)

        sample = np.random.default_rng().standard_normal(size=(N, T_query, query_dim), dtype=np.float32)

        pt_sample = torch.tensor(sample, dtype=torch.float32)
        tf_sample = tf.constant(sample)

        pt_layer = CrossAttention(query_dim=query_dim, context_dim=context_dim, heads=heads, dim_head=dim_head)
        tf_layer = TFCrossAttention(query_dim=query_dim, context_dim=context_dim, heads=heads, dim_head=dim_head)

        # init. TF weights
        _ = tf_layer(tf_sample, context=None)
        # Load PT weights
        tf_layer.base_model_prefix = ""
        tf_layer._keys_to_ignore_on_load_missing = []
        tf_layer._keys_to_ignore_on_load_unexpected = []
        load_pytorch_model_in_tf2_model(tf_layer, pt_layer, tf_inputs=tf_sample, allow_missing_keys=False)

        with torch.no_grad():
            pt_output = pt_layer(pt_sample, context=None)
        tf_output = tf_layer(tf_sample, context=None)

        max_diff = np.amax(np.abs(pt_output.numpy() - tf_output.numpy()))
        assert max_diff < 1e-6

    def test_pt_tf_with_sliced_attention(self):
        N, T_query, T_context, query_dim, context_dim, heads, dim_head = (2, 5, 3, 8, 4, 2, 8)

        sample = np.random.default_rng().standard_normal(size=(N, T_query, query_dim), dtype=np.float32)
        context = np.random.default_rng().standard_normal(size=(N, T_context, context_dim), dtype=np.float32)

        pt_sample = torch.tensor(sample, dtype=torch.float32)
        tf_sample = tf.constant(sample)

        pt_context = torch.tensor(context, dtype=torch.float32)
        tf_context = tf.constant(context)

        pt_layer = CrossAttention(query_dim=query_dim, context_dim=context_dim, heads=heads, dim_head=dim_head)
        tf_layer = TFCrossAttention(query_dim=query_dim, context_dim=context_dim, heads=heads, dim_head=dim_head)

        # Use sliced attention
        pt_layer._slice_size = 1
        tf_layer._slice_size = 1

        # init. TF weights
        _ = tf_layer(tf_sample, context=tf_context)
        # Load PT weights
        tf_layer.base_model_prefix = ""
        tf_layer._keys_to_ignore_on_load_missing = []
        tf_layer._keys_to_ignore_on_load_unexpected = []
        # TODO: Clean up
        load_pytorch_model_in_tf2_model(tf_layer, pt_layer, tf_inputs={"hidden_states": tf_sample, "context": tf_context}, allow_missing_keys=False)

        with torch.no_grad():
            pt_output = pt_layer(pt_sample, context=pt_context)
        tf_output = tf_layer(tf_sample, context=tf_context)

        max_diff = np.amax(np.abs(pt_output.numpy() - tf_output.numpy()))
        assert max_diff < 1e-6


class TFBasicTransformerBlockTest(unittest.TestCase):

    def test_pt_tf_default(self):
        N, T_query, T_context, dim, context_dim, heads, dim_head = (1, 5, 3, 8, 4, 2, 8)

        sample = np.random.default_rng().standard_normal(size=(N, T_query, dim), dtype=np.float32)
        context = np.random.default_rng().standard_normal(size=(N, T_context, context_dim), dtype=np.float32)

        pt_sample = torch.tensor(sample, dtype=torch.float32)
        tf_sample = tf.constant(sample)

        pt_context = torch.tensor(context, dtype=torch.float32)
        tf_context = tf.constant(context)

        pt_layer = BasicTransformerBlock(dim=dim, context_dim=context_dim, n_heads=heads, d_head=dim_head)
        tf_layer = TFBasicTransformerBlock(dim=dim, context_dim=context_dim, n_heads=heads, d_head=dim_head)

        # init. TF weights
        _ = tf_layer(tf_sample, context=tf_context)
        # Load PT weights
        tf_layer.base_model_prefix = ""
        tf_layer._keys_to_ignore_on_load_missing = []
        tf_layer._keys_to_ignore_on_load_unexpected = []
        # TODO: Clean up
        load_pytorch_model_in_tf2_model(tf_layer, pt_layer, tf_inputs={"hidden_states": tf_sample, "context": tf_context}, allow_missing_keys=False)

        with torch.no_grad():
            pt_output = pt_layer(pt_sample, context=pt_context)
        tf_output = tf_layer(tf_sample, context=tf_context)

        max_diff = np.amax(np.abs(pt_output.numpy() - tf_output.numpy()))
        assert max_diff < 1e-6


class TFSpatialTransformerTest(unittest.TestCase):

    def test_pt_tf_default(self):
        N, H, W, C = (1, 16, 16, 6)
        heads = 2
        dim_head = 8
        context_dim = 4
        context_seq_len = 3

        sample = np.random.default_rng().standard_normal(size=(N, C, H, W), dtype=np.float32)
        context = np.random.default_rng().standard_normal(size=(N, context_seq_len, context_dim), dtype=np.float32)

        pt_sample = torch.tensor(sample, dtype=torch.float32)
        tf_sample = tf.constant(sample)
        # (N, C, H, W) -> (N, H, W, C) for TF
        tf_sample = tf.transpose(tf.constant(sample), perm=(0, 2, 3, 1))

        pt_context = torch.tensor(context, dtype=torch.float32)
        tf_context = tf.constant(context)

        pt_layer = SpatialTransformer(in_channels=C, context_dim=context_dim, n_heads=heads, d_head=dim_head, num_groups=3)
        tf_layer = TFSpatialTransformer(in_channels=C, context_dim=context_dim, n_heads=heads, d_head=dim_head, num_groups=3)

        # init. TF weights
        _ = tf_layer(tf_sample, context=tf_context)
        # Load PT weights
        tf_layer.base_model_prefix = ""
        tf_layer._keys_to_ignore_on_load_missing = []
        tf_layer._keys_to_ignore_on_load_unexpected = []
        # TODO: Clean up
        load_pytorch_model_in_tf2_model(tf_layer, pt_layer, tf_inputs={"hidden_states": tf_sample, "context": tf_context}, allow_missing_keys=False)

        with torch.no_grad():
            pt_output = pt_layer(pt_sample, context=pt_context)
        tf_output = tf_layer(tf_sample, context=tf_context)
        # (N, H, W, C) -> (N, C, H, W)
        tf_output = tf.transpose(tf_output, perm=(0, 3, 1, 2))

        max_diff = np.amax(np.abs(pt_output.numpy() - tf_output.numpy()))
        assert max_diff < 1e-6
