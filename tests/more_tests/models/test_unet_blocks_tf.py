import numpy as np
import torch
import tensorflow as tf
import unittest

from diffusers.models.unet_blocks_tf import TFUpBlock2D, TFDownBlock2D, TFAttnDownBlock2D, TFAttnUpBlock2D, TFCrossAttnDownBlock2D, TFCrossAttnUpBlock2D, TFSkipDownBlock2D
from diffusers.models.unet_blocks import UpBlock2D, DownBlock2D, AttnDownBlock2D, AttnUpBlock2D, CrossAttnDownBlock2D, CrossAttnUpBlock2D, SkipDownBlock2D

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
        _ = tf_layer(tf_sample, temb=tf_temb)
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


class TFAttnDownBlock2DTest(unittest.TestCase):

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

        pt_layer = AttnDownBlock2D(in_channels=C, out_channels=out_C, temb_channels=temb_channels, num_layers=2, resnet_groups=C)
        tf_layer = TFAttnDownBlock2D(in_channels=C, out_channels=out_C, temb_channels=temb_channels, num_layers=2, resnet_groups=C)

        # init. TF weights
        _ = tf_layer(tf_sample, temb=tf_temb)
        # Load PT weights
        tf_layer.base_model_prefix = ""
        tf_layer._keys_to_ignore_on_load_missing = []
        tf_layer._keys_to_ignore_on_load_unexpected = []
        # TODO: Clean up
        load_pytorch_model_in_tf2_model(tf_layer, pt_layer, tf_inputs={"hidden_states": tf_sample, "temb": tf_temb}, allow_missing_keys=False)

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


class TFCrossAttnDownBlock2DTest(unittest.TestCase):

    def test_pt_tf_default(self):
        N, H, W, C = (1, 32, 32, 3)
        out_C = 2 * C
        temb_channels = 16
        encoder_seq_len = 3
        encoder_hidden_dim = 4

        sample = np.random.default_rng().standard_normal(size=(N, C, H, W), dtype=np.float32)
        temb = np.random.default_rng().standard_normal(size=(N, temb_channels), dtype=np.float32)
        encoder_hidden_states = np.random.default_rng().standard_normal(size=(N, encoder_seq_len, encoder_hidden_dim), dtype=np.float32)

        pt_sample = torch.tensor(sample)
        # (N, C, H, W) -> (N, H, W, C) for TF
        tf_sample = tf.transpose(tf.constant(sample), perm=(0, 2, 3, 1))

        pt_temb = torch.tensor(temb, dtype=torch.float32)
        tf_temb = tf.constant(temb)

        pt_encoder_hidden_states = torch.tensor(encoder_hidden_states, dtype=torch.float32)
        tf_encoder_hidden_states = tf.constant(encoder_hidden_states)

        pt_layer = CrossAttnDownBlock2D(in_channels=C, out_channels=out_C, temb_channels=temb_channels, num_layers=2, resnet_groups=C, cross_attention_dim=encoder_hidden_dim)
        tf_layer = TFCrossAttnDownBlock2D(in_channels=C, out_channels=out_C, temb_channels=temb_channels, num_layers=2, resnet_groups=C, cross_attention_dim=encoder_hidden_dim)

        # init. TF weights
        _ = tf_layer(tf_sample, temb=tf_temb, encoder_hidden_states=tf_encoder_hidden_states)
        # Load PT weights
        tf_layer.base_model_prefix = ""
        tf_layer._keys_to_ignore_on_load_missing = []
        tf_layer._keys_to_ignore_on_load_unexpected = []
        # TODO: Clean up
        load_pytorch_model_in_tf2_model(tf_layer, pt_layer, tf_inputs={"hidden_states": tf_sample, "temb": tf_temb, "encoder_hidden_states": tf_encoder_hidden_states}, allow_missing_keys=False)

        with torch.no_grad():
            pt_output = pt_layer(pt_sample, temb=pt_temb, encoder_hidden_states=pt_encoder_hidden_states)
        tf_output = tf_layer(tf_sample, temb=tf_temb, encoder_hidden_states=tf_encoder_hidden_states)

        # TODO: Use a recursive check method
        for pt_o, tf_o in zip(pt_output, tf_output):
            if not isinstance(pt_o, tuple):
                pt_o, tf_o = (pt_o,), (tf_o,)
            for _pt_o, _tf_o in zip(pt_o, tf_o):
                # (N, H, W, C) -> (N, C, H, W)
                _tf_o = tf.transpose(_tf_o, perm=(0, 3, 1, 2))
                max_diff = np.amax(np.abs(np.array(_pt_o) - np.array(_tf_o)))
                assert max_diff < 5e-6


class TFSkipDownBlock2DTest(unittest.TestCase):

    def test_pt_tf_default(self):
        N, H, W, C = (1, 32, 32, 8)
        out_C = 2 * C
        temb_channels = 16

        sample = np.random.default_rng().standard_normal(size=(N, C, H, W), dtype=np.float32)
        temb = np.random.default_rng().standard_normal(size=(N, temb_channels), dtype=np.float32)
        skip_sample = np.random.default_rng().standard_normal(size=(N, 3, H, W), dtype=np.float32)

        pt_sample = torch.tensor(sample)
        # (N, C, H, W) -> (N, H, W, C) for TF
        tf_sample = tf.transpose(tf.constant(sample), perm=(0, 2, 3, 1))

        pt_temb = torch.tensor(temb, dtype=torch.float32)
        tf_temb = tf.constant(temb)

        pt_skip_sample = torch.tensor(skip_sample, dtype=torch.float32)
        # (N, C, H, W) -> (N, H, W, C) for TF
        tf_skip_sample = tf.transpose(tf.constant(skip_sample), perm=(0, 2, 3, 1))

        pt_layer = SkipDownBlock2D(in_channels=C, out_channels=out_C, temb_channels=temb_channels, num_layers=2)
        tf_layer = TFSkipDownBlock2D(in_channels=C, out_channels=out_C, temb_channels=temb_channels, num_layers=2)

        # init. TF weights
        _ = tf_layer(tf_sample, temb=tf_temb, skip_sample=tf_skip_sample)
        # Load PT weights
        tf_layer.base_model_prefix = ""
        tf_layer._keys_to_ignore_on_load_missing = []
        tf_layer._keys_to_ignore_on_load_unexpected = []
        # TODO: Clean up
        load_pytorch_model_in_tf2_model(tf_layer, pt_layer, tf_inputs={"hidden_states": tf_sample, "temb": tf_temb, "skip_sample": tf_skip_sample}, allow_missing_keys=False)

        with torch.no_grad():
            pt_output = pt_layer(pt_sample, temb=pt_temb, skip_sample=pt_skip_sample)
        tf_output = tf_layer(tf_sample, temb=tf_temb, skip_sample=tf_skip_sample)

        # TODO: Use a recursive check method
        for pt_o, tf_o in zip(pt_output, tf_output):
            if not isinstance(pt_o, tuple):
                pt_o, tf_o = (pt_o,), (tf_o,)
            for _pt_o, _tf_o in zip(pt_o, tf_o):
                # (N, H, W, C) -> (N, C, H, W)
                _tf_o = tf.transpose(_tf_o, perm=(0, 3, 1, 2))
                max_diff = np.amax(np.abs(np.array(_pt_o) - np.array(_tf_o)))
                assert max_diff < 5e-6


class TFUpBlock2DTest(unittest.TestCase):

    def test_pt_tf_default(self):
        N, H, W, C = (1, 16, 16, 6)
        prev_out_C = 6
        out_C = 3
        temb_channels = 16

        sample = np.random.default_rng().standard_normal(size=(N, C, H, W), dtype=np.float32)
        temb = np.random.default_rng().standard_normal(size=(N, temb_channels), dtype=np.float32)

        # TODO: Try to understand what should be the correct numbers of channels.
        h_0 = np.random.default_rng().standard_normal(size=(N, C, H, W), dtype=np.float32)
        h_1 = np.random.default_rng().standard_normal(size=(N, out_C, H, W), dtype=np.float32)

        pt_sample = torch.tensor(sample)
        # (N, C, H, W) -> (N, H, W, C) for TF
        tf_sample = tf.transpose(tf.constant(sample), perm=(0, 2, 3, 1))

        pt_h_0 = torch.tensor(h_0)
        # (N, C, H, W) -> (N, H, W, C) for TF
        tf_h_0 = tf.transpose(tf.constant(h_0), perm=(0, 2, 3, 1))

        pt_h_1 = torch.tensor(h_1)
        # (N, C, H, W) -> (N, H, W, C) for TF
        tf_h_1 = tf.transpose(tf.constant(h_1), perm=(0, 2, 3, 1))

        pt_res_hidden_states_tuple = (pt_h_0, pt_h_1)
        tf_res_hidden_states_tuple = (tf_h_0, tf_h_1)

        pt_temb = torch.tensor(temb, dtype=torch.float32)
        tf_temb = tf.constant(temb)

        pt_layer = UpBlock2D(in_channels=C, prev_output_channel=prev_out_C, out_channels=out_C, temb_channels=temb_channels, num_layers=2, resnet_groups=out_C)
        tf_layer = TFUpBlock2D(in_channels=C, prev_output_channel=prev_out_C, out_channels=out_C, temb_channels=temb_channels, num_layers=2, resnet_groups=out_C)

        # init. TF weights
        _ = tf_layer(tf_sample, res_hidden_states_tuple=tf_res_hidden_states_tuple, temb=tf_temb)
        # Load PT weights
        tf_layer.base_model_prefix = ""
        tf_layer._keys_to_ignore_on_load_missing = []
        tf_layer._keys_to_ignore_on_load_unexpected = []
        # TODO: Clean up
        load_pytorch_model_in_tf2_model(tf_layer, pt_layer, tf_inputs={"hidden_states": tf_sample, "res_hidden_states_tuple": tf_res_hidden_states_tuple, "temb": tf_temb}, allow_missing_keys=False)

        with torch.no_grad():
            pt_output = pt_layer(pt_sample, res_hidden_states_tuple=pt_res_hidden_states_tuple, temb=pt_temb)
        tf_output = tf_layer(tf_sample, res_hidden_states_tuple=tf_res_hidden_states_tuple, temb=tf_temb)
        # (N, H, W, C) -> (N, C, H, W)
        tf_output = tf.transpose(tf_output, perm=(0, 3, 1, 2))

        max_diff = np.amax(np.abs(pt_output.numpy() - tf_output.numpy()))
        assert max_diff < 1e-6


class TFAttnUpBlock2DTest(unittest.TestCase):

    def test_pt_tf_default(self):
        N, H, W, C = (1, 16, 16, 6)
        prev_out_C = 6
        out_C = 3
        temb_channels = 16

        sample = np.random.default_rng().standard_normal(size=(N, C, H, W), dtype=np.float32)
        temb = np.random.default_rng().standard_normal(size=(N, temb_channels), dtype=np.float32)

        # TODO: Try to understand what should be the correct numbers of channels.
        h_0 = np.random.default_rng().standard_normal(size=(N, C, H, W), dtype=np.float32)
        h_1 = np.random.default_rng().standard_normal(size=(N, out_C, H, W), dtype=np.float32)

        pt_sample = torch.tensor(sample)
        # (N, C, H, W) -> (N, H, W, C) for TF
        tf_sample = tf.transpose(tf.constant(sample), perm=(0, 2, 3, 1))

        pt_h_0 = torch.tensor(h_0)
        # (N, C, H, W) -> (N, H, W, C) for TF
        tf_h_0 = tf.transpose(tf.constant(h_0), perm=(0, 2, 3, 1))

        pt_h_1 = torch.tensor(h_1)
        # (N, C, H, W) -> (N, H, W, C) for TF
        tf_h_1 = tf.transpose(tf.constant(h_1), perm=(0, 2, 3, 1))

        pt_res_hidden_states_tuple = (pt_h_0, pt_h_1)
        tf_res_hidden_states_tuple = (tf_h_0, tf_h_1)

        pt_temb = torch.tensor(temb, dtype=torch.float32)
        tf_temb = tf.constant(temb)

        pt_layer = AttnUpBlock2D(in_channels=C, prev_output_channel=prev_out_C, out_channels=out_C, temb_channels=temb_channels, num_layers=2, resnet_groups=out_C)
        tf_layer = TFAttnUpBlock2D(in_channels=C, prev_output_channel=prev_out_C, out_channels=out_C, temb_channels=temb_channels, num_layers=2, resnet_groups=out_C)

        # init. TF weights
        _ = tf_layer(tf_sample, res_hidden_states_tuple=tf_res_hidden_states_tuple, temb=tf_temb)
        # Load PT weights
        tf_layer.base_model_prefix = ""
        tf_layer._keys_to_ignore_on_load_missing = []
        tf_layer._keys_to_ignore_on_load_unexpected = []
        # TODO: Clean up
        load_pytorch_model_in_tf2_model(tf_layer, pt_layer, tf_inputs={"hidden_states": tf_sample, "res_hidden_states_tuple": tf_res_hidden_states_tuple, "temb": tf_temb}, allow_missing_keys=False)

        with torch.no_grad():
            pt_output = pt_layer(pt_sample, res_hidden_states_tuple=pt_res_hidden_states_tuple, temb=pt_temb)
        tf_output = tf_layer(tf_sample, res_hidden_states_tuple=tf_res_hidden_states_tuple, temb=tf_temb)
        # (N, H, W, C) -> (N, C, H, W)
        tf_output = tf.transpose(tf_output, perm=(0, 3, 1, 2))

        max_diff = np.amax(np.abs(pt_output.numpy() - tf_output.numpy()))
        assert max_diff < 1e-6


class TFCrossAttnUpBlock2DTest(unittest.TestCase):

    def test_pt_tf_default(self):
        N, H, W, C = (1, 16, 16, 6)
        prev_out_C = 6
        out_C = 3
        temb_channels = 16
        encoder_seq_len = 3
        encoder_hidden_dim = 4

        sample = np.random.default_rng().standard_normal(size=(N, C, H, W), dtype=np.float32)
        temb = np.random.default_rng().standard_normal(size=(N, temb_channels), dtype=np.float32)
        encoder_hidden_states = np.random.default_rng().standard_normal(size=(N, encoder_seq_len, encoder_hidden_dim), dtype=np.float32)

        # TODO: Try to understand what should be the correct numbers of channels.
        h_0 = np.random.default_rng().standard_normal(size=(N, C, H, W), dtype=np.float32)
        h_1 = np.random.default_rng().standard_normal(size=(N, out_C, H, W), dtype=np.float32)

        pt_sample = torch.tensor(sample)
        # (N, C, H, W) -> (N, H, W, C) for TF
        tf_sample = tf.transpose(tf.constant(sample), perm=(0, 2, 3, 1))

        pt_h_0 = torch.tensor(h_0)
        # (N, C, H, W) -> (N, H, W, C) for TF
        tf_h_0 = tf.transpose(tf.constant(h_0), perm=(0, 2, 3, 1))

        pt_h_1 = torch.tensor(h_1)
        # (N, C, H, W) -> (N, H, W, C) for TF
        tf_h_1 = tf.transpose(tf.constant(h_1), perm=(0, 2, 3, 1))

        pt_res_hidden_states_tuple = (pt_h_0, pt_h_1)
        tf_res_hidden_states_tuple = (tf_h_0, tf_h_1)

        pt_temb = torch.tensor(temb, dtype=torch.float32)
        tf_temb = tf.constant(temb)

        pt_encoder_hidden_states = torch.tensor(encoder_hidden_states, dtype=torch.float32)
        tf_encoder_hidden_states = tf.constant(encoder_hidden_states)

        pt_layer = CrossAttnUpBlock2D(in_channels=C, prev_output_channel=prev_out_C, out_channels=out_C, temb_channels=temb_channels, num_layers=2, resnet_groups=out_C, cross_attention_dim=encoder_hidden_dim)
        tf_layer = TFCrossAttnUpBlock2D(in_channels=C, prev_output_channel=prev_out_C, out_channels=out_C, temb_channels=temb_channels, num_layers=2, resnet_groups=out_C, cross_attention_dim=encoder_hidden_dim)

        # init. TF weights
        _ = tf_layer(tf_sample, res_hidden_states_tuple=tf_res_hidden_states_tuple, temb=tf_temb, encoder_hidden_states=tf_encoder_hidden_states)
        # Load PT weights
        tf_layer.base_model_prefix = ""
        tf_layer._keys_to_ignore_on_load_missing = []
        tf_layer._keys_to_ignore_on_load_unexpected = []
        # TODO: Clean up
        load_pytorch_model_in_tf2_model(tf_layer, pt_layer, tf_inputs={"hidden_states": tf_sample, "res_hidden_states_tuple": tf_res_hidden_states_tuple, "temb": tf_temb, "encoder_hidden_states": tf_encoder_hidden_states}, allow_missing_keys=False)

        with torch.no_grad():
            pt_output = pt_layer(pt_sample, res_hidden_states_tuple=pt_res_hidden_states_tuple, temb=pt_temb, encoder_hidden_states=pt_encoder_hidden_states)
        tf_output = tf_layer(tf_sample, res_hidden_states_tuple=tf_res_hidden_states_tuple, temb=tf_temb, encoder_hidden_states=tf_encoder_hidden_states)
        # (N, H, W, C) -> (N, C, H, W)
        tf_output = tf.transpose(tf_output, perm=(0, 3, 1, 2))

        max_diff = np.amax(np.abs(pt_output.numpy() - tf_output.numpy()))
        assert max_diff < 1e-6
