import numpy as np
import torch
import tensorflow as tf
import unittest

from diffusers.models.embeddings_tf import TFTimestepEmbedding
from diffusers.models.embeddings import TimestepEmbedding

from transformers import load_pytorch_model_in_tf2_model


class TFTimestepEmbeddingTest(unittest.TestCase):

    def test_pt_tf_default(self):
        N, C = (1, 16)
        time_embed_dim = 2 * C

        sample = np.random.default_rng().standard_normal(size=(N, C), dtype=np.float32)

        pt_sample = torch.tensor(sample)
        tf_sample = tf.constant(sample)

        pt_layer = TimestepEmbedding(channel=C, time_embed_dim=time_embed_dim)
        tf_layer = TFTimestepEmbedding(channel=C, time_embed_dim=time_embed_dim)

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
