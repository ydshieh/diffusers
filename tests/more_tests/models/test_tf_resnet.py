import numpy as np
import torch
import tensorflow as tf
import unittest

from diffusers.models.tf_resnet import TFUpsample2D, TFDownsample2D
from diffusers.models.resnet import Upsample2D, Downsample2D

from transformers import load_pytorch_model_in_tf2_model


class TFUpsample2DTest(unittest.TestCase):

    def test_default(self):
        tf.random.set_seed(0)
        N, H, W, C = (1, 32, 32, 32)
        sample = tf.random.normal(shape=(N, H, W, C))
        layer = TFUpsample2D(channels=32, use_conv=False)
        output = layer(sample)

        assert output.shape == (N, 2 * H, 2 * W, C)

        output_slice = output[0, -3:, -3:, -1]
        expected_slice = tf.constant(
            [
                [ 0.29245523, -1.2734733,  -1.2734733],
                [-0.47425485, -0.4724851,  -0.4724851],
                [-0.47425485, -0.4724851,  -0.4724851],
            ],
            dtype=tf.float32
        )
        max_diff = np.amax(np.abs(output_slice - expected_slice))
        assert max_diff < 1e-6

    def test_with_conv(self):
        tf.random.set_seed(0)
        N, H, W, C = (1, 32, 32, 32)
        sample = tf.random.normal(shape=(N, H, W, C))
        layer = TFUpsample2D(channels=32, use_conv=True)
        output = layer(sample)

        assert output.shape == (N, 2 * H, 2 * W, C)

        output_slice = output[0, -3:, -3:, -1]
        expected_slice = tf.constant(
            [
                [-1.749454, -0.9941745, 0.15457422],
                [-0.9837213, -0.14480323, 0.9730556],
                [-0.55730736, -1.1202476, -0.9680407],
            ],
            dtype=tf.float32
        )
        max_diff = np.amax(np.abs(output_slice - expected_slice))
        assert max_diff < 1e-6

    def test_with_conv_transpose(self):
        tf.random.set_seed(0)
        N, H, W, C = (1, 32, 32, 32)
        sample = tf.random.normal(shape=(N, H, W, C))
        layer = TFUpsample2D(channels=32, use_conv=False, use_conv_transpose=True)
        output = layer(sample)

        assert output.shape == (N, 2 * H, 2 * W, C)

        output_slice = output[0, -3:, -3:, -1]
        expected_slice = tf.constant(
            [
                [0.50377005, -0.3625132, 0.5826132],
                [0.30704838, 0.22791186, -0.15754233],
                [-0.11379892, 0.09514441, -0.14046755],
            ],
            dtype=tf.float32
        )
        max_diff = np.amax(np.abs(output_slice - expected_slice))
        assert max_diff < 1e-6

    def test_pt_tf_default(self):
        N, H, W, C = (1, 32, 32, 32)
        sample = np.random.default_rng().standard_normal(size=(N, C, H, W), dtype=np.float32)

        pt_sample = torch.tensor(sample)
        # (N, C, H, W) -> (N, H, W, C) for TF
        tf_sample = tf.transpose(tf.constant(sample), perm=(0, 2, 3, 1))

        pt_layer = Upsample2D(channels=32, use_conv=False)
        tf_layer = TFUpsample2D(channels=32, use_conv=False)

        with torch.no_grad():
            pt_output = pt_layer(pt_sample)
        tf_output = tf_layer(tf_sample)
        # (N, H, W, C) -> (N, C, H, W)
        tf_output = tf.transpose(tf_output, perm=(0, 3, 1, 2))

        max_diff = np.amax(np.abs(pt_output.numpy() - tf_output.numpy()))
        assert max_diff < 1e-6

    def test_pt_tf_with_conv(self):
        N, H, W, C = (1, 32, 32, 32)
        sample = np.random.default_rng().standard_normal(size=(N, C, H, W), dtype=np.float32)

        pt_sample = torch.tensor(sample)
        # (N, C, H, W) -> (N, H, W, C) for TF
        tf_sample = tf.transpose(tf.constant(sample), perm=(0, 2, 3, 1))

        pt_layer = Upsample2D(channels=32, use_conv=True)
        tf_layer = TFUpsample2D(channels=32, use_conv=True)

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
        # TODO: why this is particular large.
        # The values returned from `F.interpolate` and `tf.image.resize` in the corresponding layers have 0.0 as diff.
        # So the diff comes from the convolution layers. Potentially due to stride on larger input (64 * 64 instead of 32 * 32)
        assert max_diff < 5e-6

    def test_pt_tf_with_conv_transpose(self):
        N, H, W, C = (1, 32, 32, 32)
        sample = np.random.default_rng().standard_normal(size=(N, C, H, W), dtype=np.float32)

        pt_sample = torch.tensor(sample)
        # (N, C, H, W) -> (N, H, W, C) for TF
        tf_sample = tf.transpose(tf.constant(sample), perm=(0, 2, 3, 1))

        pt_layer = Upsample2D(channels=32, use_conv=False, use_conv_transpose=True)
        tf_layer = TFUpsample2D(channels=32, use_conv=False, use_conv_transpose=True)

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


class TFDownsample2DTest(unittest.TestCase):

    def test_default(self):
        tf.random.set_seed(0)
        N, H, W, C = (1, 64, 64, 32)
        sample = tf.random.normal(shape=(N, H, W, C))
        layer = TFDownsample2D(channels=32, use_conv=False)
        output = layer(sample)

        assert output.shape == (N, H // 2, W // 2, C)

        output_slice = output[0, -3:, -3:, -1]
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
        layer = TFDownsample2D(channels=32, use_conv=True)
        output = layer(sample)

        assert output.shape == (N, H // 2, W // 2, C)

        output_slice = output[0, -3:, -3:, -1]
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

        pt_layer = Downsample2D(channels=32, use_conv=False)
        tf_layer = TFDownsample2D(channels=32, use_conv=False)

        with torch.no_grad():
            pt_output = pt_layer(pt_sample)
        tf_output = tf_layer(tf_sample)
        # (N, H, W, C) -> (N, C, H, W)
        tf_output = tf.transpose(tf_output, perm=(0, 3, 1, 2))

        max_diff = np.amax(np.abs(pt_output.numpy() - tf_output.numpy()))
        assert max_diff < 1e-6

    def test_pt_tf_with_conv(self):
        N, H, W, C = (1, 16, 16, 3)
        sample = np.random.default_rng().standard_normal(size=(N, C, H, W), dtype=np.float32)

        pt_sample = torch.tensor(sample)
        # (N, C, H, W) -> (N, H, W, C) for TF
        tf_sample = tf.transpose(tf.constant(sample), perm=(0, 2, 3, 1))

        pt_layer = Downsample2D(channels=3, out_channels=4, use_conv=True)
        tf_layer = TFDownsample2D(channels=3, out_channels=4, use_conv=True)

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


class TFFirTest(unittest.TestCase):

    # TODO: remove once higher level tests pass
    def test_upfirdn2d_native(self):

        from diffusers.models.tf_resnet import upfirdn2d_native as tf_upfirdn2d_native

        tf.random.set_seed(0)
        N, H, W, C = (1, 16, 16, 3)
        K0, K1 = (3, 3)
        sample = tf.random.normal(shape=(N, H, W, C))
        kernel = tf.random.normal(shape=(K0, K1))
        up = 1
        down = 1
        pad = (0, 0)

        tf_inputs = {"input": sample, "kernel": kernel, "up": up, "down": down, "pad": pad}
        tf_inputs = {k: tf.constant(v, dtype=tf.float32) if isinstance(v, np.ndarray) else v for k, v in tf_inputs.items()}

        output = tf_upfirdn2d_native(**tf_inputs)

        output_slice = output[0, -3:, -3:, -1]
        expected_slice = tf.constant(
            [
                [-5.128561, -4.720215, 3.5403748],
                [3.5484822, -2.7563298, 0.37529594],
                [0.6663396, -2.983273, -1.3901749],
            ],
            dtype=tf.float32
        )
        max_diff = np.amax(np.abs(output_slice - expected_slice))
        assert max_diff < 1e-6

    def test_pt_tf_upfirdn2d_native(self):

        from diffusers.models.resnet import upfirdn2d_native
        from diffusers.models.tf_resnet import upfirdn2d_native as tf_upfirdn2d_native

        N, H, W, C = (1, 16, 16, 3)
        K0, K1 = (3, 3)
        sample = np.random.default_rng().standard_normal(size=(N, H, W, C), dtype=np.float32)
        kernel = np.random.default_rng().standard_normal(size=(K0, K1), dtype=np.float32)

        up_down_to_test = [(1, 1), (1, 2)]
        pads_to_test = [(0, 0), (1, 1), (1, 2)]

        for up, down in up_down_to_test:

            for pad in pads_to_test:

                pt_inputs = {"input": sample, "kernel": kernel, "up": up, "down": down, "pad": pad}
                tf_inputs = {"input": sample, "kernel": kernel, "up": up, "down": down, "pad": pad}

                pt_inputs = {k: torch.tensor(v, dtype=torch.float32) if isinstance(v, np.ndarray) else v for k, v in pt_inputs.items()}
                tf_inputs = {k: tf.constant(v, dtype=tf.float32) if isinstance(v, np.ndarray) else v for k, v in tf_inputs.items()}

                # (N, H, W, C) -> (N, C, H, W) for PT
                pt_inputs["input"] = torch.permute(pt_inputs["input"], dims=(0, 3, 1, 2))

                pt_output = upfirdn2d_native(**pt_inputs)
                tf_output = tf_upfirdn2d_native(**tf_inputs)
                # (N, H, W, C) -> (N, C, H, W)
                tf_output = tf.transpose(tf_output, perm=(0, 3, 1, 2))

                max_diff = np.amax(np.abs(pt_output.numpy() - tf_output.numpy()))
                assert max_diff < 1e-6
