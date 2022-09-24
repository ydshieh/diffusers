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
