import numpy as np
import torch
import tensorflow as tf
import unittest

from diffusers.models.attention_tf import TFGEGLU, TFFeedForward, TFAttentionBlock
from diffusers.models.attention import GEGLU, FeedForward, AttentionBlock

from transformers import load_pytorch_model_in_tf2_model


class TFGEGLUTest(unittest.TestCase):

    def test_pt_tf_default(self):
        N, C = (1, 16)
        C_out = 32

        sample = np.random.default_rng().standard_normal(size=(N, C), dtype=np.float32)

        pt_sample = torch.tensor(sample)
        # (N, C, H, W) -> (N, H, W, C) for TF
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
        # (N, C, H, W) -> (N, H, W, C) for TF
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
