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
from .resnet_tf import TFUpsample2D, TFDownsample2D, TFResnetBlock2D, TFFirDownsample2D


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
