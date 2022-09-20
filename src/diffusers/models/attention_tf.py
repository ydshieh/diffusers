from typing import Optional

import tensorflow as tf


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


class TFAttentionBlock(tf.keras.layers.Layer):
    """
    An attention block that allows spatial positions to attend to each other. Originally ported from here, but adapted
    to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    Uses three q, k, v linear layers to compute attention.

    Parameters:
        channels (:obj:`int`): The number of channels in the input and output.
        num_head_channels (:obj:`int`, *optional*):
            The number of channels in each head. If None, then `num_heads` = 1.
        num_groups (:obj:`int`, *optional*, defaults to 32): The number of groups to use for group norm.
        rescale_output_factor (:obj:`float`, *optional*, defaults to 1.0): The factor to rescale the output by.
        eps (:obj:`float`, *optional*, defaults to 1e-5): The epsilon value to use for group norm.
    """

    def __init__(
        self,
        channels: int,
        num_head_channels: Optional[int] = None,
        num_groups: int = 32,
        rescale_output_factor: float = 1.0,
        eps: float = 1e-5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.channels = channels

        self.num_heads = channels // num_head_channels if num_head_channels is not None else 1
        self.num_head_size = num_head_channels
        self.group_norm = TFGroupNormalization(groups=num_groups, epsilon=eps, center=True, scale=True, name="group_norm")

        # define q,k,v as linear layers
        self.query = tf.keras.layers.Dense(units=channels, name="query")
        self.key = tf.keras.layers.Dense(units=channels, name="key")
        self.value = tf.keras.layers.Dense(units=channels, name="value")

        self.rescale_output_factor = rescale_output_factor
        self.proj_attn = tf.keras.layers.Dense(units=channels, name="proj_attn")

    def transpose_for_scores(self, projection: tf.Tensor) -> tf.Tensor:
        new_projection_shape = tuple(projection.shape[:-1]) + (self.num_heads, -1)
        # move heads to 2nd position (B, T, H * D) -> (B, T, H, D) -> (B, H, T, D)
        new_projection = tf.transpose(tf.reshape(projection, shape=new_projection_shape), perm=(0, 2, 1, 3))
        return new_projection

    def call(self, hidden_states):
        residual = hidden_states
        batch, height, width, channel = hidden_states.shape

        # norm
        hidden_states = self.group_norm(hidden_states)

        hidden_states = tf.reshape(hidden_states, shape=(batch, height * width, channel))

        # proj to q, k, v
        query_proj = self.query(hidden_states)
        key_proj = self.key(hidden_states)
        value_proj = self.value(hidden_states)

        # transpose
        query_states = self.transpose_for_scores(query_proj)
        key_states = self.transpose_for_scores(key_proj)
        value_states = self.transpose_for_scores(value_proj)

        # get scores
        scale = 1.0 / tf.math.sqrt(tf.math.sqrt(self.channels / self.num_heads))

        attention_scores = tf.matmul(query_states * scale, tf.transpose(key_states, perm=(0, 1, 3, 2)) * scale)
        attention_probs = tf.cast(tf.math.softmax(tf.cast(attention_scores, dtype=tf.float32), axis=-1), dtype=attention_scores.dtype)

        # compute attention output
        hidden_states = tf.matmul(attention_probs, value_states)

        hidden_states = tf.transpose(hidden_states, perm=(0, 2, 1, 3))
        new_hidden_states_shape = tuple(hidden_states.shape[:-2]) + (self.channels,)
        hidden_states = tf.reshape(hidden_states, shape=new_hidden_states_shape)

        # compute next hidden_states
        hidden_states = self.proj_attn(hidden_states)
        hidden_states = tf.reshape(hidden_states, shape=(batch, height, width, channel))

        # res connect and rescale
        hidden_states = (hidden_states + residual) / self.rescale_output_factor
        return hidden_states


class TFSpatialTransformer(tf.keras.layers.Layer):
    """
    Transformer block for image-like data. First, project the input (aka embedding) and reshape to b, t, d. Then apply
    standard transformer action. Finally, reshape to image.

    Parameters:
        in_channels (:obj:`int`): The number of channels in the input and output.
        n_heads (:obj:`int`): The number of heads to use for multi-head attention.
        d_head (:obj:`int`): The number of channels in each head.
        depth (:obj:`int`, *optional*, defaults to 1): The number of layers of Transformer blocks to use.
        dropout (:obj:`float`, *optional*, defaults to 0.1): The dropout probability to use.
        context_dim (:obj:`int`, *optional*): The number of context dimensions to use.
    """

    def __init__(
        self,
        in_channels: int,
        n_heads: int,
        d_head: int,
        depth: int = 1,
        dropout: float = 0.0,
        num_groups: int = 32,
        context_dim: Optional[int] = None,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_head
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = TFGroupNormalization(groups=num_groups, epsilon=1e-6, center=True, scale=True, name="norm")

        self.proj_in = tf.keras.layers.Conv2D(filters=inner_dim, kernel_size=1, strides=1, padding="VALID", name="proj_in")

        self.transformer_blocks = []
        for d in range(depth):
            self.transformer_blocks.append(
                TFBasicTransformerBlock(
                    dim=inner_dim,
                    n_heads=n_heads,
                    d_head=d_head,
                    dropout=dropout,
                    context_dim=context_dim,
                    name=f"transformer_blocks_._{d}"
                )
            )

        self.proj_out = tf.keras.layers.Conv2D(filters=in_channels, kernel_size=1, strides=1, padding="VALID", name="proj_out")

    def _set_attention_slice(self, slice_size):
        for block in self.transformer_blocks:
            block._set_attention_slice(slice_size)

    def call(self, hidden_states, context=None):
        # TODO: Remove
        if isinstance(hidden_states, dict):
            hidden_states, context = hidden_states["hidden_states"], hidden_states["context"]

        # note: if no context is given, cross-attention defaults to self-attention
        batch, height, weight, channel = hidden_states.shape
        residual = hidden_states
        hidden_states = self.norm(hidden_states)
        hidden_states = self.proj_in(hidden_states)
        hidden_states = tf.reshape(hidden_states, shape=(batch, height * weight, -1))
        for block in self.transformer_blocks:
            hidden_states = block(hidden_states, context=context)
        hidden_states = tf.reshape(hidden_states, shape=(batch, height, weight, -1))
        hidden_states = self.proj_out(hidden_states)
        return hidden_states + residual


class TFBasicTransformerBlock(tf.keras.layers.Layer):
    r"""
    A basic Transformer block.

    Parameters:
        dim (:obj:`int`): The number of channels in the input and output.
        n_heads (:obj:`int`): The number of heads to use for multi-head attention.
        d_head (:obj:`int`): The number of channels in each head.
        dropout (:obj:`float`, *optional*, defaults to 0.0): The dropout probability to use.
        context_dim (:obj:`int`, *optional*): The size of the context vector for cross attention.
        gated_ff (:obj:`bool`, *optional*, defaults to :obj:`False`): Whether to use a gated feed-forward network.
        checkpoint (:obj:`bool`, *optional*, defaults to :obj:`False`): Whether to use checkpointing.
    """

    def __init__(
        self,
        dim: int,
        n_heads: int,
        d_head: int,
        dropout=0.0,
        context_dim: Optional[int] = None,
        gated_ff: bool = True,
        checkpoint: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.attn1 = TFCrossAttention(
            query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout, name="attn1"
        )  # is a self-attention
        self.ff = TFFeedForward(dim, dropout=dropout, glu=gated_ff, name="ff")
        self.attn2 = TFCrossAttention(
            query_dim=dim, context_dim=context_dim, heads=n_heads, dim_head=d_head, dropout=dropout, name="attn2"
        )  # is self-attn if context is none
        self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-05, name="norm1")
        self.norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-05, name="norm2")
        self.norm3 = tf.keras.layers.LayerNormalization(epsilon=1e-05, name="norm3")
        # TODO: Remove
        self.checkpoint = checkpoint

    def _set_attention_slice(self, slice_size):
        self.attn1._slice_size = slice_size
        self.attn2._slice_size = slice_size

    def call(self, hidden_states, context=None):
        # TODO: Remove
        if isinstance(hidden_states, dict):
            hidden_states, context = hidden_states["hidden_states"], hidden_states["context"]

        hidden_states = self.attn1(self.norm1(hidden_states)) + hidden_states
        hidden_states = self.attn2(self.norm2(hidden_states), context=context) + hidden_states
        hidden_states = self.ff(self.norm3(hidden_states)) + hidden_states
        return hidden_states


class TFCrossAttention(tf.keras.layers.Layer):
    r"""
    A cross attention layer.

    Parameters:
        query_dim (:obj:`int`): The number of channels in the query.
        context_dim (:obj:`int`, *optional*):
            The number of channels in the context. If not given, defaults to `query_dim`.
        heads (:obj:`int`,  *optional*, defaults to 8): The number of heads to use for multi-head attention.
        dim_head (:obj:`int`,  *optional*, defaults to 64): The number of channels in each head.
        dropout (:obj:`float`, *optional*, defaults to 0.0): The dropout probability to use.
    """

    def __init__(
        self, query_dim: int, context_dim: Optional[int] = None, heads: int = 8, dim_head: int = 64, dropout: int = 0.0, **kwargs
    ):
        super().__init__(**kwargs)
        inner_dim = dim_head * heads
        context_dim = context_dim if context_dim is not None else query_dim

        self.scale = dim_head**-0.5
        self.heads = heads
        # for slice_size > 0 the attention score computation
        # is split across the batch axis to save memory
        # You can set slice_size with `set_attention_slice`
        self._slice_size = None

        self.to_q = tf.keras.layers.Dense(units=inner_dim, use_bias=False, name="to_q")
        self.to_k = tf.keras.layers.Dense(units=inner_dim, use_bias=False, name="to_k")
        self.to_v = tf.keras.layers.Dense(units=inner_dim, use_bias=False, name="to_v")

        output_layer = tf.keras.layers.Dense(query_dim, name="to_out_._0")
        dropout_layer = tf.keras.layers.Dropout(dropout)
        self.to_out = [output_layer, dropout_layer]

    def reshape_heads_to_batch_dim(self, tensor):
        batch_size, seq_len, dim = tensor.shape
        head_size = self.heads
        tensor = tf.reshape(tensor, shape=(batch_size, seq_len, head_size, dim // head_size))
        tensor = tf.reshape(tf.transpose(tensor, perm=(0, 2, 1, 3)), shape=(batch_size * head_size, seq_len, dim // head_size))
        return tensor

    def reshape_batch_dim_to_heads(self, tensor):
        batch_size, seq_len, dim = tensor.shape
        head_size = self.heads
        tensor = tf.reshape(tensor, shape=(batch_size // head_size, head_size, seq_len, dim))
        tensor = tf.reshape(tf.transpose(tensor, perm=(0, 2, 1, 3)), shape=(batch_size // head_size, seq_len, dim * head_size))
        return tensor

    def call(self, hidden_states, context=None, mask=None):
        # TODO: Remove
        if isinstance(hidden_states, dict):
            hidden_states, context = hidden_states["hidden_states"], hidden_states["context"]

        batch_size, sequence_length, dim = hidden_states.shape

        query = self.to_q(hidden_states)
        context = context if context is not None else hidden_states
        key = self.to_k(context)
        value = self.to_v(context)

        query = self.reshape_heads_to_batch_dim(query)
        key = self.reshape_heads_to_batch_dim(key)
        value = self.reshape_heads_to_batch_dim(value)

        # TODO(PVP) - mask is currently never used. Remember to re-implement when used

        # attention, what we cannot get enough of

        if self._slice_size is None or query.shape[0] // self._slice_size == 1:
            hidden_states = self._attention(query, key, value)
        else:
            hidden_states = self._sliced_attention(query, key, value, sequence_length, dim)

        for layer in self.to_out:
            hidden_states = layer(hidden_states)

        return hidden_states

    def _attention(self, query, key, value):
        attention_scores = tf.matmul(query, tf.transpose(key, perm=(0, 2, 1))) * self.scale
        attention_probs = tf.math.softmax(attention_scores, axis=-1)
        # compute attention output
        hidden_states = tf.matmul(attention_probs, value)
        # reshape hidden_states
        hidden_states = self.reshape_batch_dim_to_heads(hidden_states)
        return hidden_states

    def _sliced_attention(self, query, key, value, sequence_length, dim):
        batch_size_attention = query.shape[0]
        # TODO: Use a buffer
        all_attn_slice = []

        slice_size = self._slice_size if self._slice_size is not None else batch_size_attention
        for i in range(batch_size_attention // slice_size):
            start_idx = i * slice_size
            end_idx = (i + 1) * slice_size
            attn_slice = tf.matmul(query[start_idx:end_idx], tf.transpose(key[start_idx:end_idx], perm=(0, 2, 1))) * self.scale
            attn_slice = tf.math.softmax(attn_slice, axis=-1)
            attn_slice = tf.matmul(attn_slice, value[start_idx:end_idx])

            all_attn_slice.append(attn_slice)

        hidden_states = tf.concat(all_attn_slice, axis=0)

        # reshape hidden_states
        hidden_states = self.reshape_batch_dim_to_heads(hidden_states)
        return hidden_states


class TFFeedForward(tf.keras.layers.Layer):
    r"""
    A feed-forward layer.

    Parameters:
        dim (:obj:`int`): The number of channels in the input.
        dim_out (:obj:`int`, *optional*): The number of channels in the output. If not given, defaults to `dim`.
        mult (:obj:`int`, *optional*, defaults to 4): The multiplier to use for the hidden dimension.
        glu (:obj:`bool`, *optional*, defaults to :obj:`False`): Whether to use GLU activation.
        dropout (:obj:`float`, *optional*, defaults to 0.0): The dropout probability to use.
    """

    def __init__(
        self, dim: int, dim_out: Optional[int] = None, mult: int = 4, glu: bool = False, dropout: float = 0.0, **kwargs
    ):
        super().__init__(**kwargs)
        inner_dim = int(dim * mult)
        dim_out = dim_out if dim_out is not None else dim
        project_in = TFGEGLU(dim, inner_dim, name="net_._0")
        dropout_layer = tf.keras.layers.Dropout(dropout)
        dense_layer = tf.keras.layers.Dense(units=dim_out, name="net_._2")

        self.net = [project_in, dropout_layer, dense_layer]

    def call(self, hidden_states):
        for layer in self.net:
            hidden_states = layer(hidden_states)
        return hidden_states


# feedforward
class TFGEGLU(tf.keras.layers.Layer):
    r"""
    A variant of the gated linear unit activation function from https://arxiv.org/abs/2002.05202.

    Parameters:
        dim_in (:obj:`int`): The number of channels in the input.
        dim_out (:obj:`int`): The number of channels in the output.
    """

    def __init__(self, dim_in: int, dim_out: int, **kwargs):
        super().__init__(**kwargs)
        self.proj = tf.keras.layers.Dense(units=dim_out * 2, name="proj")

    def call(self, hidden_states):
        hidden_states = self.proj(hidden_states)
        # TODO: `GEGLU` uses `torch.Tensor.chunk`, which might have a last chunk with smaller size, or even returns less
        # than the specified number of chunks. We should check the inputs and raise errors in this case.
        hidden_states, gate = tf.split(hidden_states, num_or_size_splits=2, axis=-1)

        return hidden_states * tf.keras.activations.gelu(gate)
