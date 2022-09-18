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
    ):
        super().__init__()
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
