import itertools
import numpy as np
import torch
import tensorflow as tf
import unittest

from diffusers.models.tf_resnet import TFUpsample2D, TFDownsample2D, TFFirUpsample2D, TFFirDownsample2D, TFGroupNormalization, TFResnetBlock2D
from diffusers.models.resnet import Upsample2D, Downsample2D, FirUpsample2D, FirDownsample2D, ResnetBlock2D

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
        sample = np.random.default_rng().standard_normal(size=(N, C, H, W), dtype=np.float32)
        kernel = np.random.default_rng().standard_normal(size=(K0, K1), dtype=np.float32)

        up_down_to_test = [(1, 1), (1, 2)]
        pads_to_test = [(0, 0), (1, 1), (1, 2)]

        for up, down in up_down_to_test:

            for pad in pads_to_test:

                pt_inputs = {"input": sample, "kernel": kernel, "up": up, "down": down, "pad": pad}
                tf_inputs = {"input": sample, "kernel": kernel, "up": up, "down": down, "pad": pad}

                pt_inputs = {k: torch.tensor(v, dtype=torch.float32) if isinstance(v, np.ndarray) else v for k, v in pt_inputs.items()}
                tf_inputs = {k: tf.constant(v, dtype=tf.float32) if isinstance(v, np.ndarray) else v for k, v in tf_inputs.items()}

                # (N, C, H, W) -> (N, H, W, C) for TF
                tf_inputs["input"] = tf.transpose(tf_inputs["input"], perm=(0, 2, 3, 1))

                pt_output = upfirdn2d_native(**pt_inputs)
                tf_output = tf_upfirdn2d_native(**tf_inputs)
                # (N, H, W, C) -> (N, C, H, W)
                tf_output = tf.transpose(tf_output, perm=(0, 3, 1, 2))

                max_diff = np.amax(np.abs(pt_output.numpy() - tf_output.numpy()))
                assert max_diff < 1e-6

    def test_upsample_2d(self):

        from diffusers.models.tf_resnet import upsample_2d as tf_upsample_2d

        tf.random.set_seed(0)
        N, H, W, C = (1, 16, 16, 3)
        K0, K1 = (3, 3)
        sample = tf.random.normal(shape=(N, H, W, C))
        kernel = tf.random.normal(shape=(K0, K1))

        # check with defaults
        output = tf_upsample_2d(sample, kernel=None)
        assert output.shape == (N, 2 * H, 2 * W, C)
        output_slice = output[0, -3:, -3:, -1]
        expected_slice = tf.constant(
            [
                [-0.4011506, -0.7577537, -0.7577537],
                [-1.0255308, 1.2520727, 1.2520727],
                [-1.0255308, 1.2520727, 1.2520727],
            ],
            dtype=tf.float32
        )
        max_diff = np.amax(np.abs(output_slice - expected_slice))
        assert max_diff < 1e-6

        # check with factor = 3
        factor = 3
        output = tf_upsample_2d(sample, kernel=None, factor=factor)
        assert output.shape == (N, factor * H, factor * W, C)
        output_slice = output[0, -3:, -3:, -1]
        expected_slice = tf.constant(
            [
                [1.2520727, 1.2520727, 1.2520727],
                [1.2520727, 1.2520727, 1.2520727],
                [1.2520727, 1.2520727, 1.2520727],
            ],
            dtype=tf.float32
        )
        max_diff = np.amax(np.abs(output_slice - expected_slice))
        assert max_diff < 1e-6

        # check a slice near the center
        output_slice = output[0, 23:26, 23:26, -1]
        expected_slice = tf.constant(
            [
                [1.2157538, -0.52501285, -0.52501285],
                [0.5482155, 1.0518552, 1.0518552],
                [0.5482155, 1.0518552, 1.0518552]
            ],
            dtype=tf.float32
        )
        max_diff = np.amax(np.abs(output_slice - expected_slice))
        assert max_diff < 1e-6

        # check with explicit `kernel`
        output = tf_upsample_2d(sample, kernel=kernel)
        assert output.shape == (N, 2 * H, 2 * W, C)
        output_slice = output[0, -3:, -3:, -1]
        expected_slice = tf.constant(
            [
                # [0.58139217, 2.442034, 1.0982211],
                # [-8.306607, 25.57224, -10.625638],
                # [1.4863136, 8.635653, -1.8146434],
                [0.5813927, 2.4420362, 1.0982219],
                [-8.306613, 25.572262, -10.625643],
                [1.4863148, 8.635658, -1.8146449],
            ],
            dtype=tf.float32
        )
        max_diff = np.amax(np.abs(output_slice - expected_slice))
        assert max_diff < 1e-6

    def test_pt_tf_upsample_2d(self):

        from diffusers.models.resnet import upsample_2d
        from diffusers.models.tf_resnet import upsample_2d as tf_upsample_2d

        N, H, W, C = (1, 16, 16, 3)
        K0, K1 = (3, 3)
        sample = np.random.default_rng().standard_normal(size=(N, C, H, W), dtype=np.float32)
        kernel = np.random.default_rng().standard_normal(size=(K0, K1), dtype=np.float32)

        pt_sample = torch.tensor(sample)
        # (N, C, H, W) -> (N, H, W, C) for TF
        tf_sample = tf.transpose(tf.constant(sample), perm=(0, 2, 3, 1))

        pt_kernel = kernel
        tf_kernel = tf.constant(kernel)

        # check with defaults
        pt_output = upsample_2d(pt_sample, kernel=None)
        tf_output = tf_upsample_2d(tf_sample, kernel=None)
        # (N, H, W, C) -> (N, C, H, W)
        tf_output = tf.transpose(tf_output, perm=(0, 3, 1, 2))
        max_diff = np.amax(np.abs(pt_output.numpy() - tf_output.numpy()))
        assert max_diff < 1e-6

        # check with factor = 3
        pt_output = upsample_2d(pt_sample, kernel=None, factor=3)
        tf_output = tf_upsample_2d(tf_sample, kernel=None, factor=3)
        # (N, H, W, C) -> (N, C, H, W)
        tf_output = tf.transpose(tf_output, perm=(0, 3, 1, 2))
        max_diff = np.amax(np.abs(pt_output.numpy() - tf_output.numpy()))
        assert max_diff < 1e-6

        # check with explicit `kernel`
        pt_output = upsample_2d(pt_sample, kernel=pt_kernel)
        tf_output = tf_upsample_2d(tf_sample, kernel=tf_kernel)
        # (N, H, W, C) -> (N, C, H, W)
        tf_output = tf.transpose(tf_output, perm=(0, 3, 1, 2))
        max_diff = np.amax(np.abs(pt_output.numpy() - tf_output.numpy()))
        assert max_diff < 1e-6

    def test_downsample_2d(self):

        from diffusers.models.tf_resnet import downsample_2d as tf_downsample_2d

        tf.random.set_seed(0)
        N, H, W, C = (1, 32, 32, 3)
        K0, K1 = (3, 3)
        sample = tf.random.normal(shape=(N, H, W, C))
        kernel = tf.random.normal(shape=(K0, K1))

        # check with defaults
        output = tf_downsample_2d(sample, kernel=None)
        assert output.shape == (N, H // 2, W // 2, C)
        output_slice = output[0, -3:, -3:, -1]
        expected_slice = tf.constant(
            [
                [-0.01057184, -0.1050401, 0.10744253],
                [ 0.5084425, 0.25627035, -0.877445],
                [-0.18831787, -0.14461315, 0.13966256],
            ],
            dtype=tf.float32
        )
        max_diff = np.amax(np.abs(output_slice - expected_slice))
        assert max_diff < 1e-6

        # check with factor = 3
        factor = 3
        output = tf_downsample_2d(sample, kernel=None, factor=factor)
        assert output.shape == (N, H // factor, W // factor, C)
        output_slice = output[0, -3:, -3:, -1]
        expected_slice = tf.constant(
            [
                [3.7484446e-01, -2.5557938e-01, -6.5304033e-02],
                [3.7692755e-01, 3.7176535e-05, 1.3446024e-01],
                [-2.0504236e-01, 9.0178549e-02, 2.8782129e-01],
            ],
            dtype=tf.float32
        )
        max_diff = np.amax(np.abs(output_slice - expected_slice))
        assert max_diff < 1e-6

        # check with explicit `kernel`
        output = tf_downsample_2d(sample, kernel=kernel)
        assert output.shape == (N, H // 2, W // 2, C)
        output_slice = output[0, -3:, -3:, -1]
        expected_slice = tf.constant(
            [
                [-5.600157, 7.580162, -7.9246755],
                [10.4908085, -1.410429, 1.7640928],
                [-2.453312, -3.5338497, 0.4914587],
            ],
            dtype=tf.float32
        )
        max_diff = np.amax(np.abs(output_slice - expected_slice))
        assert max_diff < 1e-6

    def test_pt_tf_downsample_2d(self):

        from diffusers.models.resnet import downsample_2d
        from diffusers.models.tf_resnet import downsample_2d as tf_downsample_2d

        N, H, W, C = (1, 32, 32, 3)
        K0, K1 = (3, 3)
        sample = np.random.default_rng().standard_normal(size=(N, C, H, W), dtype=np.float32)
        kernel = np.random.default_rng().standard_normal(size=(K0, K1), dtype=np.float32)

        pt_sample = torch.tensor(sample)
        # (N, C, H, W) -> (N, H, W, C) for TF
        tf_sample = tf.transpose(tf.constant(sample), perm=(0, 2, 3, 1))

        pt_kernel = kernel
        tf_kernel = tf.constant(kernel)

        # check with defaults
        pt_output = downsample_2d(pt_sample, kernel=None)
        tf_output = tf_downsample_2d(tf_sample, kernel=None)
        # (N, H, W, C) -> (N, C, H, W)
        tf_output = tf.transpose(tf_output, perm=(0, 3, 1, 2))
        max_diff = np.amax(np.abs(pt_output.numpy() - tf_output.numpy()))
        assert max_diff < 1e-6

        # check with factor =downsample_2d 3
        pt_output = downsample_2d(pt_sample, kernel=None, factor=3)
        tf_output = tf_downsample_2d(tf_sample, kernel=None, factor=3)
        # (N, H, W, C) -> (N, C, H, W)
        tf_output = tf.transpose(tf_output, perm=(0, 3, 1, 2))
        max_diff = np.amax(np.abs(pt_output.numpy() - tf_output.numpy()))
        assert max_diff < 1e-6

        # check with explicit `kernel`
        pt_output = downsample_2d(pt_sample, kernel=pt_kernel)
        tf_output = tf_downsample_2d(tf_sample, kernel=tf_kernel)
        # (N, H, W, C) -> (N, C, H, W)
        tf_output = tf.transpose(tf_output, perm=(0, 3, 1, 2))
        max_diff = np.amax(np.abs(pt_output.numpy() - tf_output.numpy()))
        assert max_diff < 1e-6


class TFFirUpsample2DTest(unittest.TestCase):

    def test_default(self):
        tf.random.set_seed(0)
        N, H, W, C = (1, 16, 16, 3)
        sample = tf.random.normal(shape=(N, H, W, C))
        layer = TFFirUpsample2D(use_conv=False)
        output = layer(sample)

        assert output.shape == (N, 2 * H, 2 * W, C)

        output_slice = output[0, -3:, -3:, -1]
        expected_slice = tf.constant(
            [
                [-0.48175848, -0.33078414, -0.19147274],
                [-0.46467274, 0.3448532, 0.5622121],
                [-0.3420974, 0.5120039, 0.70429087],
            ],
            dtype=tf.float32
        )
        max_diff = np.amax(np.abs(output_slice - expected_slice))
        assert max_diff < 1e-6

    def test_with_conv(self):
        tf.random.set_seed(0)
        N, H, W, C = (1, 16, 16, 3)
        out_C = 5
        sample = tf.random.normal(shape=(N, H, W, C))
        layer = TFFirUpsample2D(out_channels=out_C, use_conv=True)
        output = layer(sample)

        assert output.shape == (N, 2 * H, 2 * W, out_C)

        output_slice = output[0, -3:, -3:, -1]
        expected_slice = tf.constant(
            [
                [-0.5779555, -0.46531677, 0.01123914],
                [-0.18219061, -0.21603137, 0.11068934],
                [0.27656645, 0.06588552, -0.04983421],
            ],
            dtype=tf.float32
        )
        max_diff = np.amax(np.abs(output_slice - expected_slice))
        assert max_diff < 1e-6

    def test_pt_tf_default(self):
        N, H, W, C = (1, 16, 16, 3)
        out_C = C
        sample = np.random.normal(size=(N, C, H, W))

        pt_sample = torch.tensor(sample)
        # (N, C, H, W) -> (N, H, W, C) for TF
        tf_sample = tf.transpose(tf.constant(sample), perm=(0, 2, 3, 1))

        pt_layer = FirUpsample2D(channels=C, out_channels=out_C, use_conv=False)
        tf_layer = TFFirUpsample2D(out_channels=out_C, use_conv=False)

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

    def test_pt_tf_default(self):
        N, H, W, C = (1, 16, 16, 3)
        out_C = 5
        sample = np.random.normal(size=(N, C, H, W))

        pt_sample = torch.tensor(sample, dtype=torch.float32)
        # (N, C, H, W) -> (N, H, W, C) for TF
        tf_sample = tf.transpose(tf.constant(sample), perm=(0, 2, 3, 1))

        pt_layer = FirUpsample2D(channels=C, out_channels=out_C, use_conv=True)
        tf_layer = TFFirUpsample2D(out_channels=out_C, use_conv=True)

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


class TFFirDownsample2DTest(unittest.TestCase):

    def test_default(self):
        tf.random.set_seed(0)
        N, H, W, C = (1, 32, 32, 3)
        sample = tf.random.normal(shape=(N, H, W, C))
        layer = TFFirDownsample2D(use_conv=False)
        output = layer(sample)

        assert output.shape == (N, H // 2, W // 2, C)

        output_slice = output[0, -3:, -3:, -1]
        expected_slice = tf.constant(
            [
                [-0.12315781, -0.11423445, 0.00152066],
                [0.25104553, 0.10429873, -0.4266413],
                [-0.02045484, 0.01432847, 0.07339813],
            ],
            dtype=tf.float32
        )
        max_diff = np.amax(np.abs(output_slice - expected_slice))
        assert max_diff < 1e-6

    def test_with_conv(self):
        tf.random.set_seed(0)
        N, H, W, C = (1, 32, 32, 3)
        out_C = 5
        sample = tf.random.normal(shape=(N, H, W, C))
        layer = TFFirDownsample2D(out_channels=out_C, use_conv=True)
        output = layer(sample)

        assert output.shape == (N, H // 2, W // 2, out_C)

        output_slice = output[0, -3:, -3:, -1]
        expected_slice = tf.constant(
            [
                [0.14284192, -0.02286262, 0.13295601],
                [0.21965423, -0.22582096, -0.13607837],
                [-0.07344813, -0.19338597, -0.1325784 ],
            ],
            dtype=tf.float32
        )
        max_diff = np.amax(np.abs(output_slice - expected_slice))
        assert max_diff < 1e-6

    def test_pt_tf_default(self):
        N, H, W, C = (1, 32, 32, 3)
        out_C = C
        sample = np.random.normal(size=(N, C, H, W))

        pt_sample = torch.tensor(sample)
        # (N, C, H, W) -> (N, H, W, C) for TF
        tf_sample = tf.transpose(tf.constant(sample), perm=(0, 2, 3, 1))

        pt_layer = FirDownsample2D(channels=C, out_channels=out_C, use_conv=False)
        tf_layer = TFFirDownsample2D(out_channels=out_C, use_conv=False)

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

    def test_pt_tf_default(self):
        N, H, W, C = (1, 32, 32, 3)
        out_C = 5
        sample = np.random.normal(size=(N, C, H, W))

        pt_sample = torch.tensor(sample, dtype=torch.float32)
        # (N, C, H, W) -> (N, H, W, C) for TF
        tf_sample = tf.transpose(tf.constant(sample), perm=(0, 2, 3, 1))

        pt_layer = FirDownsample2D(channels=C, out_channels=out_C, use_conv=True)
        tf_layer = TFFirDownsample2D(out_channels=out_C, use_conv=True)

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


class TFGroupNormalizationTest(unittest.TestCase):

    def test_default(self):
        tf.random.set_seed(0)
        N, H, W, C = (1, 16, 16, 4)
        sample = tf.random.normal(shape=(N, H, W, C))

        all_num_groups = [1, C // 2, C]
        for groups in all_num_groups:

            layer = TFGroupNormalization(groups=groups, epsilon=1e-5)
            output = layer(sample)

            assert output.shape == (N, H, W, C)

            if groups == C:
                output_slice = output[0, -3:, -3:, -1]
                expected_slice = tf.constant(
                    [
                        [-1.8358996, 0.54756016, -0.00880986],
                        [0.9807491, 0.11368423, -1.4105656],
                        [-0.43764484, -0.5612552, -0.81363803],
                    ],
                    dtype=tf.float32
                )
                max_diff = np.amax(np.abs(output_slice - expected_slice))
                assert max_diff < 1e-6

    def test_pt_tf_default(self):

        from torch.nn import GroupNorm

        N, H, W, C = (1, 16, 16, 4)
        sample = np.random.normal(size=(N, C, H, W))

        pt_sample = torch.tensor(sample, dtype=torch.float32)
        # (N, C, H, W) -> (N, H, W, C) for TF
        tf_sample = tf.transpose(tf.constant(sample), perm=(0, 2, 3, 1))

        all_num_groups = [1, C // 2, C]
        for groups in all_num_groups:

            pt_layer = GroupNorm(num_groups=groups, num_channels=C, eps=1e-5)
            tf_layer = TFGroupNormalization(groups=groups, epsilon=1e-5)

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


class TFResnetBlock2DTest(unittest.TestCase):

    def test_pt_tf_default(self):

        N, H, W, C = (1, 16, 16, 4)
        out_C = 6
        sample = np.random.normal(size=(N, C, H, W))

        pt_sample = torch.tensor(sample, dtype=torch.float32)
        # (N, C, H, W) -> (N, H, W, C) for TF
        tf_sample = tf.transpose(tf.constant(sample), perm=(0, 2, 3, 1))

        all_options = {
            "in_channels": [C],
            "out_channels": [None, C, out_C],
            "conv_shortcut": [False, True],
            "temb_channels": [None, 8],
            "groups": [1, C // 2, C],
            "non_linearity": ["swish", "mish", "silu"],
            # TODO: Add different values once available
            "time_embedding_norm": ["default"],
            # TODO: Add "sde_vp" once available in TF
            "kernel": [None, "fir", "sde_vp"],
            "use_in_shortcut": [None, False, True],
            "up": [False, True],
            "down": [False, True],
        }
        keys = list(all_options.keys())
        values = [all_options[k] for k in keys]

        for options in itertools.product(*values):

            configs = {k: v for k, v in zip(keys, options)}
            out_channels = configs["out_channels"]
            groups = configs["groups"]

            # This gives an error for both PT/TF
            # TODO: Better error message
            if configs["use_in_shortcut"] is False and configs["out_channels"] != configs["in_channels"]:
                continue

            # depends on `out_channels`
            if out_channels is not None:
                if out_channels % groups == 0:
                    all_groups_out = [None, 1, groups, out_channels // 2, out_channels]
                else:
                    all_groups_out = [1, out_channels // 2, out_channels]
            else:
                all_groups_out = [None, 1, C // 2, C]

            for groups_out in all_groups_out:

                temb = None

                pt_layer = ResnetBlock2D(**configs, groups_out=groups_out)
                tf_layer = TFResnetBlock2D(**configs, groups_out=groups_out)

                # init. TF weights
                _ = tf_layer(tf_sample, temb=temb)
                # Load PT weights
                tf_layer.base_model_prefix = ""
                tf_layer._keys_to_ignore_on_load_missing = []
                tf_layer._keys_to_ignore_on_load_unexpected = []
                load_pytorch_model_in_tf2_model(tf_layer, pt_layer, tf_inputs=tf_sample, allow_missing_keys=False)

                with torch.no_grad():
                    pt_output = pt_layer(pt_sample, temb=temb)
                tf_output = tf_layer(tf_sample, temb=temb)

                # (N, H, W, C) -> (N, C, H, W)
                tf_output = tf.transpose(tf_output, perm=(0, 3, 1, 2))

                max_diff = np.amax(np.abs(pt_output.numpy() - tf_output.numpy()))
                assert max_diff < 5e-6
