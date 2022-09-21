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


import numpy as np
import tensorflow as tf

from .attention_tf import TFAttentionBlock, TFSpatialTransformer
from .resnet_tf import TFUpsample2D, TFDownsample2D, TFResnetBlock2D, TFFirDownsample2D, TFFirUpsample2D


class TFGroupNormalization(tf.keras.layers.Layer):
    """Group normalization layer.

    From tensorflow-addons https://www.tensorflow.org/addons/api_docs/python/tfa/layers/GroupNormalization
    """

    def __init__(
        self,
        groups: int = 32,
        axis: int = -1,
        epsilon: float = 1e-3,
        center: bool = True,
        scale: bool = True,
        beta_initializer: tf.keras.initializers.Initializer = "zeros",
        gamma_initializer: tf.keras.initializers.Initializer = "ones",
        beta_regularizer: tf.keras.regularizers.Regularizer = None,
        gamma_regularizer: tf.keras.regularizers.Regularizer = None,
        beta_constraint: tf.keras.constraints.Constraint = None,
        gamma_constraint: tf.keras.constraints.Constraint = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.supports_masking = True
        self.groups = groups
        self.axis = axis
        self.epsilon = epsilon
        self.center = center
        self.scale = scale
        self.beta_initializer = tf.keras.initializers.get(beta_initializer)
        self.gamma_initializer = tf.keras.initializers.get(gamma_initializer)
        self.beta_regularizer = tf.keras.regularizers.get(beta_regularizer)
        self.gamma_regularizer = tf.keras.regularizers.get(gamma_regularizer)
        self.beta_constraint = tf.keras.constraints.get(beta_constraint)
        self.gamma_constraint = tf.keras.constraints.get(gamma_constraint)
        self._check_axis()

    def build(self, input_shape):

        self._check_if_input_shape_is_none(input_shape)
        self._set_number_of_groups_for_instance_norm(input_shape)
        self._check_size_of_dimensions(input_shape)
        self._create_input_spec(input_shape)

        self._add_gamma_weight(input_shape)
        self._add_beta_weight(input_shape)
        self.built = True
        super().build(input_shape)

    def call(self, inputs):

        input_shape = tf.keras.backend.int_shape(inputs)
        tensor_input_shape = tf.shape(inputs)

        reshaped_inputs, group_shape = self._reshape_into_groups(
            inputs, input_shape, tensor_input_shape
        )

        normalized_inputs = self._apply_normalization(reshaped_inputs, input_shape)

        is_instance_norm = (input_shape[self.axis] // self.groups) == 1
        if not is_instance_norm:
            outputs = tf.reshape(normalized_inputs, tensor_input_shape)
        else:
            outputs = normalized_inputs

        return outputs

    def get_config(self):
        config = {
            "groups": self.groups,
            "axis": self.axis,
            "epsilon": self.epsilon,
            "center": self.center,
            "scale": self.scale,
            "beta_initializer": tf.keras.initializers.serialize(self.beta_initializer),
            "gamma_initializer": tf.keras.initializers.serialize(
                self.gamma_initializer
            ),
            "beta_regularizer": tf.keras.regularizers.serialize(self.beta_regularizer),
            "gamma_regularizer": tf.keras.regularizers.serialize(
                self.gamma_regularizer
            ),
            "beta_constraint": tf.keras.constraints.serialize(self.beta_constraint),
            "gamma_constraint": tf.keras.constraints.serialize(self.gamma_constraint),
        }
        base_config = super().get_config()
        return {**base_config, **config}

    def _reshape_into_groups(self, inputs, input_shape, tensor_input_shape):

        group_shape = [tensor_input_shape[i] for i in range(len(input_shape))]
        is_instance_norm = (input_shape[self.axis] // self.groups) == 1
        if not is_instance_norm:
            group_shape[self.axis] = input_shape[self.axis] // self.groups
            group_shape.insert(self.axis, self.groups)
            group_shape = tf.stack(group_shape)
            reshaped_inputs = tf.reshape(inputs, group_shape)
            return reshaped_inputs, group_shape
        else:
            return inputs, group_shape

    def _apply_normalization(self, reshaped_inputs, input_shape):

        group_shape = tf.keras.backend.int_shape(reshaped_inputs)
        group_reduction_axes = list(range(1, len(group_shape)))
        is_instance_norm = (input_shape[self.axis] // self.groups) == 1
        if not is_instance_norm:
            axis = -2 if self.axis == -1 else self.axis - 1
        else:
            axis = -1 if self.axis == -1 else self.axis - 1
        group_reduction_axes.pop(axis)

        mean, variance = tf.nn.moments(
            reshaped_inputs, group_reduction_axes, keepdims=True
        )

        gamma, beta = self._get_reshaped_weights(input_shape)
        normalized_inputs = tf.nn.batch_normalization(
            reshaped_inputs,
            mean=mean,
            variance=variance,
            scale=gamma,
            offset=beta,
            variance_epsilon=self.epsilon,
        )
        return normalized_inputs

    def _get_reshaped_weights(self, input_shape):
        broadcast_shape = self._create_broadcast_shape(input_shape)
        gamma = None
        beta = None
        if self.scale:
            gamma = tf.reshape(self.gamma, broadcast_shape)

        if self.center:
            beta = tf.reshape(self.beta, broadcast_shape)
        return gamma, beta

    def _check_if_input_shape_is_none(self, input_shape):
        dim = input_shape[self.axis]
        if dim is None:
            raise ValueError(
                "Axis " + str(self.axis) + " of "
                "input tensor should have a defined dimension "
                "but the layer received an input with shape " + str(input_shape) + "."
            )

    def _set_number_of_groups_for_instance_norm(self, input_shape):
        dim = input_shape[self.axis]

        if self.groups == -1:
            self.groups = dim

    def _check_size_of_dimensions(self, input_shape):

        dim = input_shape[self.axis]
        if dim < self.groups:
            raise ValueError(
                "Number of groups (" + str(self.groups) + ") cannot be "
                "more than the number of channels (" + str(dim) + ")."
            )

        if dim % self.groups != 0:
            raise ValueError(
                "Number of groups (" + str(self.groups) + ") must be a "
                "multiple of the number of channels (" + str(dim) + ")."
            )

    def _check_axis(self):

        if self.axis == 0:
            raise ValueError(
                "You are trying to normalize your batch axis. Do you want to "
                "use tf.layer.batch_normalization instead"
            )

    def _create_input_spec(self, input_shape):

        dim = input_shape[self.axis]
        self.input_spec = tf.keras.layers.InputSpec(
            ndim=len(input_shape), axes={self.axis: dim}
        )

    def _add_gamma_weight(self, input_shape):

        dim = input_shape[self.axis]
        shape = (dim,)

        if self.scale:
            self.gamma = self.add_weight(
                shape=shape,
                name="gamma",
                initializer=self.gamma_initializer,
                regularizer=self.gamma_regularizer,
                constraint=self.gamma_constraint,
            )
        else:
            self.gamma = None

    def _add_beta_weight(self, input_shape):

        dim = input_shape[self.axis]
        shape = (dim,)

        if self.center:
            self.beta = self.add_weight(
                shape=shape,
                name="beta",
                initializer=self.beta_initializer,
                regularizer=self.beta_regularizer,
                constraint=self.beta_constraint,
            )
        else:
            self.beta = None

    def _create_broadcast_shape(self, input_shape):
        broadcast_shape = [1] * len(input_shape)
        is_instance_norm = (input_shape[self.axis] // self.groups) == 1
        if not is_instance_norm:
            broadcast_shape[self.axis] = input_shape[self.axis] // self.groups
            broadcast_shape.insert(self.axis, self.groups)
        else:
            broadcast_shape[self.axis] = self.groups
        return broadcast_shape


class TFDownBlock2D(tf.keras.layers.Layer):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        output_scale_factor=1.0,
        add_downsample=True,
        downsample_padding=1,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.resnets = []
        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            self.resnets.append(
                TFResnetBlock2D(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                    name=f"resnets_._{i}",
                )
            )

        if add_downsample:
            self.downsamplers = [TFDownsample2D(channels=in_channels, use_conv=True, out_channels=out_channels, padding=downsample_padding, name=f"downsamplers_._{0}")]
        else:
            self.downsamplers = None

    def call(self, hidden_states, temb=None):
        output_states = ()

        for resnet in self.resnets:
            hidden_states = resnet(hidden_states, temb)
            output_states += (hidden_states,)

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)

            output_states += (hidden_states,)

        return hidden_states, output_states


class TFAttnDownBlock2D(tf.keras.layers.Layer):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        attn_num_head_channels=1,
        attention_type="default",
        output_scale_factor=1.0,
        downsample_padding=1,
        add_downsample=True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.resnets = []
        self.attentions = []

        self.attention_type = attention_type

        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            self.resnets.append(
                TFResnetBlock2D(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                    name=f"resnets_._{i}",
                )
            )
            self.attentions.append(
                TFAttentionBlock(
                    out_channels,
                    num_head_channels=attn_num_head_channels,
                    rescale_output_factor=output_scale_factor,
                    eps=resnet_eps,
                    num_groups=resnet_groups,
                    name=f"attentions_._{i}",
                )
            )

        if add_downsample:
            self.downsamplers = [TFDownsample2D(channels=in_channels, use_conv=True, out_channels=out_channels, padding=downsample_padding, name=f"downsamplers_._{0}")]
        else:
            self.downsamplers = None

    def call(self, hidden_states, temb=None):
        # TODO: Remove
        if isinstance(hidden_states, dict):
            hidden_states, temb = hidden_states["hidden_states"], hidden_states["temb"]

        output_states = ()

        for resnet, attn in zip(self.resnets, self.attentions):
            hidden_states = resnet(hidden_states, temb)
            hidden_states = attn(hidden_states)
            output_states += (hidden_states,)

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)

            output_states += (hidden_states,)

        return hidden_states, output_states


class TFCrossAttnDownBlock2D(tf.keras.layers.Layer):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        attn_num_head_channels=1,
        cross_attention_dim=1280,
        attention_type="default",
        output_scale_factor=1.0,
        downsample_padding=1,
        add_downsample=True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.resnets = []
        self.attentions = []

        self.attention_type = attention_type
        self.attn_num_head_channels = attn_num_head_channels

        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            self.resnets.append(
                TFResnetBlock2D(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                    name=f"resnets_._{i}",
                )
            )
            self.attentions.append(
                TFSpatialTransformer(
                    out_channels,
                    attn_num_head_channels,
                    out_channels // attn_num_head_channels,
                    depth=1,
                    context_dim=cross_attention_dim,
                    num_groups=resnet_groups,
                    name=f"attentions_._{i}",
                )
            )

        if add_downsample:
            self.downsamplers = [TFDownsample2D(channels=in_channels, use_conv=True, out_channels=out_channels, padding=downsample_padding, name=f"downsamplers_._{0}")]
        else:
            self.downsamplers = None

    def set_attention_slice(self, slice_size):
        if slice_size is not None and self.attn_num_head_channels % slice_size != 0:
            raise ValueError(
                f"Make sure slice_size {slice_size} is a divisor of "
                f"the number of heads used in cross_attention {self.attn_num_head_channels}"
            )
        if slice_size is not None and slice_size > self.attn_num_head_channels:
            raise ValueError(
                f"Chunk_size {slice_size} has to be smaller or equal to "
                f"the number of heads used in cross_attention {self.attn_num_head_channels}"
            )

        for attn in self.attentions:
            attn._set_attention_slice(slice_size)

    def call(self, hidden_states, temb=None, encoder_hidden_states=None):
        # TODO: Remove
        if isinstance(hidden_states, dict):
            hidden_states, temb, encoder_hidden_states = hidden_states["hidden_states"], hidden_states["temb"], hidden_states["encoder_hidden_states"]

        output_states = ()

        for resnet, attn in zip(self.resnets, self.attentions):
            hidden_states = resnet(hidden_states, temb)
            hidden_states = attn(hidden_states, context=encoder_hidden_states)
            output_states += (hidden_states,)

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)

            output_states += (hidden_states,)

        return hidden_states, output_states


class TFSkipDownBlock2D(tf.keras.layers.Layer):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_pre_norm: bool = True,
        output_scale_factor=np.sqrt(2.0),
        add_downsample=True,
        downsample_padding=1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.resnets = []

        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            self.resnets.append(
                TFResnetBlock2D(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=min(in_channels // 4, 32),
                    groups_out=min(out_channels // 4, 32),
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                    name=f"resnets_._{i}",
                )
            )

        if add_downsample:
            self.resnet_down = TFResnetBlock2D(
                in_channels=out_channels,
                out_channels=out_channels,
                temb_channels=temb_channels,
                eps=resnet_eps,
                groups=min(out_channels // 4, 32),
                dropout=dropout,
                time_embedding_norm=resnet_time_scale_shift,
                non_linearity=resnet_act_fn,
                output_scale_factor=output_scale_factor,
                pre_norm=resnet_pre_norm,
                use_in_shortcut=True,
                down=True,
                kernel="fir",
                name="resnet_down",
            )
            self.downsamplers = [TFFirDownsample2D(channels=in_channels, out_channels=out_channels, name=f"downsamplers_._{0}")]
            self.skip_conv = tf.keras.layers.Conv2D(out_channels, kernel_size=(1, 1), strides=(1, 1), name="skip_conv")
        else:
            self.resnet_down = None
            self.downsamplers = None
            self.skip_conv = None

    def call(self, hidden_states, temb=None, skip_sample=None):
        # TODO: Remove
        if isinstance(hidden_states, dict):
            hidden_states, temb, skip_sample = hidden_states["hidden_states"], hidden_states["temb"], hidden_states["skip_sample"]

        output_states = ()

        for resnet in self.resnets:
            hidden_states = resnet(hidden_states, temb)
            output_states += (hidden_states,)

        if self.downsamplers is not None:
            hidden_states = self.resnet_down(hidden_states, temb)
            for downsampler in self.downsamplers:
                skip_sample = downsampler(skip_sample)

            hidden_states = self.skip_conv(skip_sample) + hidden_states

            output_states += (hidden_states,)

        return hidden_states, output_states, skip_sample


class TFDownEncoderBlock2D(tf.keras.layers.Layer):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        output_scale_factor=1.0,
        add_downsample=True,
        downsample_padding=1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.resnets = []

        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            self.resnets.append(
                TFResnetBlock2D(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    temb_channels=None,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                    name=f"resnets_._{i}",
                )
            )

        if add_downsample:
            self.downsamplers = [TFDownsample2D(channels=in_channels, use_conv=True, out_channels=out_channels, padding=downsample_padding, name=f"downsamplers_._{0}")]
        else:
            self.downsamplers = None

    def call(self, hidden_states):
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states, temb=None)

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)

        return hidden_states


class TFUpBlock2D(tf.keras.layers.Layer):

    def __init__(
        self,
        in_channels: int,
        prev_output_channel: int,
        out_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        output_scale_factor=1.0,
        add_upsample=True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.resnets = []
        for i in range(num_layers):
            res_skip_channels = in_channels if (i == num_layers - 1) else out_channels
            resnet_in_channels = prev_output_channel if i == 0 else out_channels

            self.resnets.append(
                TFResnetBlock2D(
                    in_channels=resnet_in_channels + res_skip_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                    name=f"resnets_._{i}",
                )
            )

        if add_upsample:
            self.upsamplers = [TFUpsample2D(channels=out_channels, use_conv=True, out_channels=out_channels, name=f"upsamplers_._{0}")]
        else:
            self.upsamplers = None

    # TODO: Avoid `None` for `res_hidden_states_tuple`.
    def call(self, hidden_states, res_hidden_states_tuple=None, temb=None):
        # TODO: Remove
        if isinstance(hidden_states, dict):
            hidden_states, res_hidden_states_tuple, temb = hidden_states["hidden_states"], hidden_states["res_hidden_states_tuple"], hidden_states["temb"]

        for resnet in self.resnets:
            # pop res hidden states
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]
            hidden_states = tf.concat([hidden_states, res_hidden_states], axis=-1)

            hidden_states = resnet(hidden_states, temb)

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states)

        return hidden_states


class TFAttnUpBlock2D(tf.keras.layers.Layer):
    def __init__(
        self,
        in_channels: int,
        prev_output_channel: int,
        out_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        attention_type="default",
        attn_num_head_channels=1,
        output_scale_factor=1.0,
        add_upsample=True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.resnets = []
        self.attentions = []

        self.attention_type = attention_type

        for i in range(num_layers):
            res_skip_channels = in_channels if (i == num_layers - 1) else out_channels
            resnet_in_channels = prev_output_channel if i == 0 else out_channels

            self.resnets.append(
                TFResnetBlock2D(
                    in_channels=resnet_in_channels + res_skip_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                    name=f"resnets_._{i}",
                )
            )
            self.attentions.append(
                TFAttentionBlock(
                    out_channels,
                    num_head_channels=attn_num_head_channels,
                    rescale_output_factor=output_scale_factor,
                    eps=resnet_eps,
                    num_groups=resnet_groups,
                    name=f"attentions_._{i}",
                )
            )

        if add_upsample:
            self.upsamplers = [TFUpsample2D(channels=out_channels, use_conv=True, out_channels=out_channels, name=f"upsamplers_._{0}")]
        else:
            self.upsamplers = None

    # TODO: Avoid `None` for `res_hidden_states_tuple`.
    def call(self, hidden_states, res_hidden_states_tuple=None, temb=None):
        # TODO: Remove
        if isinstance(hidden_states, dict):
            hidden_states, res_hidden_states_tuple, temb = hidden_states["hidden_states"], hidden_states["res_hidden_states_tuple"], hidden_states["temb"]

        for resnet, attn in zip(self.resnets, self.attentions):
            # pop res hidden states
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]
            hidden_states = tf.concat([hidden_states, res_hidden_states], axis=-1)

            hidden_states = resnet(hidden_states, temb)
            hidden_states = attn(hidden_states)

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states)

        return hidden_states


class TFCrossAttnUpBlock2D(tf.keras.layers.Layer):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        prev_output_channel: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        attn_num_head_channels=1,
        cross_attention_dim=1280,
        attention_type="default",
        output_scale_factor=1.0,
        downsample_padding=1,
        add_upsample=True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.resnets = []
        self.attentions = []

        self.attention_type = attention_type
        self.attn_num_head_channels = attn_num_head_channels

        for i in range(num_layers):
            res_skip_channels = in_channels if (i == num_layers - 1) else out_channels
            resnet_in_channels = prev_output_channel if i == 0 else out_channels

            self.resnets.append(
                TFResnetBlock2D(
                    in_channels=resnet_in_channels + res_skip_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                    name=f"resnets_._{i}",
                )
            )
            self.attentions.append(
                TFSpatialTransformer(
                    out_channels,
                    attn_num_head_channels,
                    out_channels // attn_num_head_channels,
                    depth=1,
                    context_dim=cross_attention_dim,
                    num_groups=resnet_groups,
                    name=f"attentions_._{i}",
                )
            )

        if add_upsample:
            self.upsamplers = [TFUpsample2D(channels=out_channels, use_conv=True, out_channels=out_channels, name=f"upsamplers_._{0}")]
        else:
            self.upsamplers = None

    def set_attention_slice(self, slice_size):
        if slice_size is not None and self.attn_num_head_channels % slice_size != 0:
            raise ValueError(
                f"Make sure slice_size {slice_size} is a divisor of "
                f"the number of heads used in cross_attention {self.attn_num_head_channels}"
            )
        if slice_size is not None and slice_size > self.attn_num_head_channels:
            raise ValueError(
                f"Chunk_size {slice_size} has to be smaller or equal to "
                f"the number of heads used in cross_attention {self.attn_num_head_channels}"
            )

        for attn in self.attentions:
            attn._set_attention_slice(slice_size)

    # TODO: Avoid `None` for `res_hidden_states_tuple`.
    def call(self, hidden_states, res_hidden_states_tuple=None, temb=None, encoder_hidden_states=None):
        # TODO: Remove
        if isinstance(hidden_states, dict):
            hidden_states, res_hidden_states_tuple, temb, encoder_hidden_states = hidden_states["hidden_states"], hidden_states["res_hidden_states_tuple"], hidden_states["temb"], hidden_states["encoder_hidden_states"]

        for resnet, attn in zip(self.resnets, self.attentions):
            # pop res hidden states
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]
            hidden_states = tf.concat([hidden_states, res_hidden_states], axis=-1)

            hidden_states = resnet(hidden_states, temb)
            hidden_states = attn(hidden_states, context=encoder_hidden_states)

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states)

        return hidden_states


class TFSkipUpBlock2D(tf.keras.layers.Layer):
    def __init__(
        self,
        in_channels: int,
        prev_output_channel: int,
        out_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_pre_norm: bool = True,
        output_scale_factor=np.sqrt(2.0),
        add_upsample=True,
        upsample_padding=1,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.resnets = []

        for i in range(num_layers):
            res_skip_channels = in_channels if (i == num_layers - 1) else out_channels
            resnet_in_channels = prev_output_channel if i == 0 else out_channels

            self.resnets.append(
                TFResnetBlock2D(
                    in_channels=resnet_in_channels + res_skip_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=min((resnet_in_channels + res_skip_channels) // 4, 32),
                    groups_out=min(out_channels // 4, 32),
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                    name=f"resnets_._{i}",
                )
            )

        self.upsampler = TFFirUpsample2D(in_channels, out_channels=out_channels, name="upsampler")
        if add_upsample:
            self.resnet_up = TFResnetBlock2D(
                in_channels=out_channels,
                out_channels=out_channels,
                temb_channels=temb_channels,
                eps=resnet_eps,
                groups=min(out_channels // 4, 32),
                groups_out=min(out_channels // 4, 32),
                dropout=dropout,
                time_embedding_norm=resnet_time_scale_shift,
                non_linearity=resnet_act_fn,
                output_scale_factor=output_scale_factor,
                pre_norm=resnet_pre_norm,
                use_in_shortcut=True,
                up=True,
                kernel="fir",
                name="resnet_up",
            )
            # Need manual padding `1` in self.call
            self.skip_conv = tf.keras.layers.Conv2D(3, kernel_size=(3, 3), strides=(1, 1), padding="VALID", name="skip_conv")
            self.skip_norm = TFGroupNormalization(
                groups=min(out_channels // 4, 32), epsilon=resnet_eps, center=True, scale=True, name="skip_norm"
            )
            self.act = tf.keras.activations.swish
        else:
            self.resnet_up = None
            self.skip_conv = None
            self.skip_norm = None
            self.act = None

    # TODO: Avoid `None` for `res_hidden_states_tuple`.
    def call(self, hidden_states, res_hidden_states_tuple=None, temb=None, skip_sample=None):
        # TODO: Remove
        if isinstance(hidden_states, dict):
            hidden_states, res_hidden_states_tuple, temb, skip_sample = hidden_states["hidden_states"], hidden_states["res_hidden_states_tuple"], hidden_states["temb"], hidden_states["skip_sample"]

        for resnet in self.resnets:
            # pop res hidden states
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]
            hidden_states = tf.concat([hidden_states, res_hidden_states], axis=-1)

            hidden_states = resnet(hidden_states, temb)

        if skip_sample is not None:
            skip_sample = self.upsampler(skip_sample)
        else:
            skip_sample = 0

        if self.resnet_up is not None:
            skip_sample_states = self.skip_norm(hidden_states)
            skip_sample_states = self.act(skip_sample_states)
            # Pad with `1` element
            paddings = [(0, 0), (1, 1), (1, 1), (0, 0)]
            skip_sample_states = tf.pad(skip_sample_states, paddings, mode='CONSTANT', constant_values=0)
            skip_sample_states = self.skip_conv(skip_sample_states)

            skip_sample = skip_sample + skip_sample_states

            hidden_states = self.resnet_up(hidden_states, temb)

        return hidden_states, skip_sample


class TFUpDecoderBlock2D(tf.keras.layers.Layer):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        output_scale_factor=1.0,
        add_upsample=True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.resnets = []

        for i in range(num_layers):
            input_channels = in_channels if i == 0 else out_channels

            self.resnets.append(
                TFResnetBlock2D(
                    in_channels=input_channels,
                    out_channels=out_channels,
                    temb_channels=None,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                    name=f"resnets_._{i}",
                )
            )

        if add_upsample:
            self.upsamplers = [TFUpsample2D(out_channels, use_conv=True, out_channels=out_channels, name="upsamplers_._0")]
        else:
            self.upsamplers = None

    def call(self, hidden_states):
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states, temb=None)

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states)

        return hidden_states
