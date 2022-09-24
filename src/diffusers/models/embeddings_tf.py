# Copyright 2022 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import math

import numpy as np
import tensorflow as tf


def get_timestep_embedding(
    timesteps: tf.Tensor,
    embedding_dim: int,
    flip_sin_to_cos: bool = False,
    downscale_freq_shift: float = 1,
    scale: float = 1,
    max_period: int = 10000,
):
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models: Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param embedding_dim: the dimension of the output. :param max_period: controls the minimum frequency of the
    embeddings. :return: an [N x dim] Tensor of positional embeddings.
    """
    assert len(timesteps.shape) == 1, "Timesteps should be a 1d-array"

    half_dim = embedding_dim // 2
    exponent = -tf.math.log(tf.constant(max_period, dtype=tf.float32)) * tf.range(start=0, limit=half_dim, dtype=tf.float32)
    exponent = exponent / (half_dim - downscale_freq_shift)

    emb = tf.math.exp(exponent)
    emb = tf.cast(timesteps[:, tf.newaxis], dtype=tf.float32) * emb[tf.newaxis, :]

    # scale embeddings
    emb = scale * emb

    # concat sine and cosine embeddings
    emb = tf.concat([tf.math.sin(emb), tf.math.cos(emb)], axis=-1)

    # flip sine and cosine embeddings
    if flip_sin_to_cos:
        emb = tf.concat([emb[:, half_dim:], emb[:, :half_dim]], axis=-1)

    # zero pad
    if embedding_dim % 2 == 1:
        # pad on the ending side in the last dimension
        paddings = tf.constant([[0, 0], [0, 1]])
        emb = tf.pad(emb, paddings=paddings)
    return emb


class TFTimestepEmbedding(tf.keras.layers.Layer):
    def __init__(self, channel: int, time_embed_dim: int, act_fn: str = "silu", **kwargs):
        super().__init__(**kwargs)

        self.linear_1 = tf.keras.layers.Dense(units=time_embed_dim, name="linear_1")
        self.act = None
        # TODO: (Question) What's the difference between "swish" and "silu"?
        if act_fn == "silu":
            self.act = tf.keras.activations.swish
        self.linear_2 = tf.keras.layers.Dense(units=time_embed_dim, name="linear_2")

    def call(self, sample):
        sample = self.linear_1(sample)

        if self.act is not None:
            sample = self.act(sample)

        sample = self.linear_2(sample)
        return sample


class TFTimesteps(tf.keras.layers.Layer):
    def __init__(self, num_channels: int, flip_sin_to_cos: bool, downscale_freq_shift: float, **kwargs):
        super().__init__(**kwargs)
        self.num_channels = num_channels
        self.flip_sin_to_cos = flip_sin_to_cos
        self.downscale_freq_shift = downscale_freq_shift

    def call(self, timesteps):
        t_emb = get_timestep_embedding(
            timesteps,
            self.num_channels,
            flip_sin_to_cos=self.flip_sin_to_cos,
            downscale_freq_shift=self.downscale_freq_shift,
        )
        return t_emb
