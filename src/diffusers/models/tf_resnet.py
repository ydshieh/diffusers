import tensorflow as tf


class TFDownsample2D(tf.keras.layers.Layer):
    """
    A downsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs. :param use_conv: a bool determining if a convolution is
    applied. :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv=False, padding=1, out_channels=None, name="conv"):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.padding = padding
        stride = 2

        layer_name = "conv"
        # TODO: clean up after PyTorch side is done
        if name == "Conv2d_0":
            layer_name = "Conv2d_0"

        if use_conv:
            conv = tf.keras.layers.Conv2D(self.out_channels, kernel_size=3, strides=stride, padding="valid", name=layer_name)
        else:
            assert self.channels == self.out_channels
            conv = tf.keras.layers.AveragePooling2D(
                pool_size=stride,
                strides=stride,
                padding='valid',
                name=layer_name,
            )

        self.conv = conv

    def call(self, x):
        assert x.shape[-1] == self.channels
        height_pad = width_pad = (0, 0)
        if self.use_conv:
            if self.padding > 0:
                height_pad = width_pad = (self.padding, self.padding)
            else:
                # pad on the right & bottom
                height_pad = width_pad = (0, 1)
        paddings = [(0, 0), height_pad, width_pad, (0, 0)]

        x = tf.pad(x, paddings, mode='CONSTANT', constant_values=0)

        assert x.shape[-1] == self.channels
        x = self.conv(x)
        assert x.shape[-1] == self.out_channels

        return x
