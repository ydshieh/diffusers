from typing import Optional

import tensorflow as tf


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
