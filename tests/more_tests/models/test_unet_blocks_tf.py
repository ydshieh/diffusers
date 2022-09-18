import numpy as np
import torch
import tensorflow as tf
import unittest

from diffusers.models.unet_blocks_tf import TFDownBlock2D
from diffusers.models.unet_blocks import DownBlock2D

from transformers import load_pytorch_model_in_tf2_model


class TFDownBlock2DTest(unittest.TestCase):

    def test_pt_tf_default(self):
        N, H, W, C = (1, 32, 32, 3)
        out_C = 2 * C
        temb_channels = 16

        sample = np.random.default_rng().standard_normal(size=(N, C, H, W), dtype=np.float32)
        temb = np.random.default_rng().standard_normal(size=(N, temb_channels), dtype=np.float32)

        pt_sample = torch.tensor(sample)
        # (N, C, H, W) -> (N, H, W, C) for TF
        tf_sample = tf.transpose(tf.constant(sample), perm=(0, 2, 3, 1))

        pt_temb = torch.tensor(temb, dtype=torch.float32)
        tf_temb = tf.constant(temb)

        pt_layer = DownBlock2D(in_channels=C, out_channels=out_C, temb_channels=temb_channels, num_layers=2, resnet_groups=C)
        tf_layer = TFDownBlock2D(in_channels=C, out_channels=out_C, temb_channels=temb_channels, num_layers=2, resnet_groups=C)

        # init. TF weights
        _ = tf_layer(tf_sample, temb=temb)
        # Load PT weights
        tf_layer.base_model_prefix = ""
        tf_layer._keys_to_ignore_on_load_missing = []
        tf_layer._keys_to_ignore_on_load_unexpected = []
        load_pytorch_model_in_tf2_model(tf_layer, pt_layer, tf_inputs=tf_sample, allow_missing_keys=False)

        with torch.no_grad():
            pt_output = pt_layer(pt_sample, temb=pt_temb)
        tf_output = tf_layer(tf_sample, temb=tf_temb)

        # TODO: Use a recursive check method
        for pt_o, tf_o in zip(pt_output, tf_output):
            if not isinstance(pt_o, tuple):
                pt_o, tf_o = (pt_o,), (tf_o,)
            for _pt_o, _tf_o in zip(pt_o, tf_o):
                # (N, H, W, C) -> (N, C, H, W)
                _tf_o = tf.transpose(_tf_o, perm=(0, 3, 1, 2))
                max_diff = np.amax(np.abs(np.array(_pt_o) - np.array(_tf_o)))
                assert max_diff < 5e-6
