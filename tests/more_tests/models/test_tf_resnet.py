import numpy as np
import torch
import tensorflow as tf
import unittest

from diffusers.models.tf_resnet import TFDownsample2D
from diffusers.models.resnet import Downsample2D

from transformers import load_pytorch_model_in_tf2_model


class TFDownsample2DTest(unittest.TestCase):

    def test_default(self):
        tf.random.set_seed(0)
        N, H, W, C = (1, 64, 64, 32)
        sample = tf.random.normal(shape=(N, H, W, C))
        downsample = TFDownsample2D(channels=32, use_conv=False)
        downsampled = downsample(sample)

        assert downsampled.shape == (N, H // 2, W // 2, C)

        output_slice = downsampled[0, -3:, -3:, -1]
        expected_slice = tf.constant(
            [
                [0.2681359, 0.23771279, -0.40026015],
                [-0.08018474, 0.23550907, -0.20984799],
                [-0.4344082, 0.07018943, -0.8420039],
            ],
            dtype=tf.float32
        )
        max_diff = np.amax(np.abs(output_slice - expected_slice))
        assert max_diff < 1e-6

    def test_with_conv(self):
        tf.random.set_seed(0)
        N, H, W, C = (1, 64, 64, 32)
        sample = tf.random.normal(shape=(N, H, W, C))
        downsample = TFDownsample2D(channels=32, use_conv=True)
        downsampled = downsample(sample)

        assert downsampled.shape == (N, H // 2, W // 2, C)

        output_slice = downsampled[0, -3:, -3:, -1]
        expected_slice = tf.constant(
            [
                [0.26708013, -1.2414322, 1.7002869],
                [0.3099873, 0.1957025, -1.4698032],
                [1.2376794, 2.3319128, -0.6908339],
            ],
            dtype=tf.float32
        )
        max_diff = np.amax(np.abs(output_slice - expected_slice))
        assert max_diff < 1e-6

    def test_pt_tf_default(self):
        N, H, W, C = (1, 32, 32, 32)
        sample = np.random.normal(size=(N, C, H, W))

        pt_sample = torch.tensor(sample)
        # (N, C, H, W) -> (N, H, W, C) for TF
        tf_sample = tf.transpose(tf.constant(sample), perm=(0, 2, 3, 1))

        pt_downsample = Downsample2D(channels=32, use_conv=False)
        tf_downsample = TFDownsample2D(channels=32, use_conv=False)

        with torch.no_grad():
            pt_downsampled = pt_downsample(pt_sample)
        tf_downsampled = tf_downsample(tf_sample)
        # (N, H, W, C) -> (N, C, H, W)
        tf_downsampled = tf.transpose(tf_downsampled, perm=(0, 3, 1, 2))

        max_diff = np.amax(np.abs(pt_downsampled.numpy() - tf_downsampled.numpy()))
        assert max_diff < 1e-6

    def test_pt_tf_with_conv(self):
        N, H, W, C = (1, 16, 16, 3)
        sample = np.random.default_rng().standard_normal(size=(N, C, H, W), dtype=np.float32)

        pt_sample = torch.tensor(sample)
        # (N, C, H, W) -> (N, H, W, C) for TF
        tf_sample = tf.transpose(tf.constant(sample), perm=(0, 2, 3, 1))

        pt_downsample = Downsample2D(channels=3, out_channels=4, use_conv=True)
        tf_downsample = TFDownsample2D(channels=3, out_channels=4, use_conv=True)

        # init. TF weights
        _ = tf_downsample(tf_sample)
        # Load PT weights
        tf_downsample.base_model_prefix = ""
        tf_downsample._keys_to_ignore_on_load_missing = []
        tf_downsample._keys_to_ignore_on_load_unexpected = []
        load_pytorch_model_in_tf2_model(tf_downsample, pt_downsample, tf_inputs=tf_sample, allow_missing_keys=False)

        with torch.no_grad():
            pt_downsampled = pt_downsample(pt_sample)
        tf_downsampled = tf_downsample(tf_sample)
        # (N, H, W, C) -> (N, C, H, W)
        tf_downsampled = tf.transpose(tf_downsampled, perm=(0, 3, 1, 2))

        max_diff = np.amax(np.abs(pt_downsampled.numpy() - tf_downsampled.numpy()))
        assert max_diff < 1e-6
