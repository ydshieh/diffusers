import tensorflow as tf


class SimpleConv2DTransposeWithExplicitPadding(tf.keras.layers.Layer):
    """Padding in `tf.keras.layers.Conv2DTranspose` or `tf.nn.conv2d_transpose` is special, and manual padding
    before calling these layers (for explicit padding) will give different result from that given by specifying
    padding in these layers.

    Therefore, this custom layer uses `tf.nn.conv2d_transpose`, and pass explicit padding to it.
    """

    def __init__(
        self,
        filters,
        kernel_size,
        strides,
        padding,
        use_bias=True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.use_bias = use_bias

    def build(self, input_shape):

        (height, weight, in_channels) = input_shape[1:]

        # `tf.nn.conv2d_transpose` requires `kernel` to have shape `(H, W, C_out, C_in]`.
        self.kernel = self.add_weight(
            shape=(self.kernel_size[0], self.kernel_size[1], self.filters, in_channels),
            initializer="glorot_uniform",
            name="kernel",
        )

        if self.use_bias:
            self.bias = self.add_weight(
                shape=(self.filters,),
                initializer="zeros",
                name="bias",
            )

    def call(self, x):

        N, H, W, C = tf.shape(x)

        new_H = (H - 1) * self.strides[0] + self.kernel_size[0] - 2 * self.padding[0]
        new_W = (W - 1) * self.strides[1] + self.kernel_size[1] - 2 * self.padding[1]

        output_shape = (N, new_H, new_W, self.filters)

        h = tf.nn.conv2d_transpose(
            input=x,
            filters=self.kernel,
            output_shape=output_shape,
            strides=self.strides,
            padding=[(0, 0), (self.padding[0], self.padding[0]), (self.padding[1], self.padding[1]), (0, 0)]
         )

        h = h + self.bias

        return h


class TFUpsample2D(tf.keras.layers.Layer):
    """
    An upsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs. :param use_conv: a bool determining if a convolution is
    applied. :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 upsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv=False, use_conv_transpose=False, out_channels=None, _conv_name="conv"):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_conv_transpose = use_conv_transpose

        # TODO: clean up after PyTorch side is done
        if _conv_name != "conv":
            _conv_name = "Conv2d_0"

        conv = None
        if use_conv_transpose:
            # `Conv2DTranspose` requires different hacks (than `Conv2DTranspose`) for explicit padding number.
            # (We can't pad manually before calling this layer, as the result will be different)
            conv = SimpleConv2DTransposeWithExplicitPadding(filters=self.out_channels, kernel_size=(4, 4), strides=(2, 2), padding=(1, 1), name=_conv_name)
        elif use_conv:
            # Need to pad manually in call() before calling self.conv() for explicit padding number.
            conv = tf.keras.layers.Conv2D(self.out_channels, kernel_size=3, padding="valid", name=_conv_name)

        # TODO: clean up after PyTorch side is done
        # setattr(self, _conv_name, conv)
        self.conv = conv
        self._conv_name = _conv_name

    def call(self, x):
        assert x.shape[1] == self.channels
        if self.use_conv_transpose:
            # padding is performed in `SimpleConv2DTransposeWithExplicitPadding`
            return self.conv(x)

        # TODO: Q: dynamic or static here?
        (height, width) = tf.shape(x)[1:3]
        x = tf.image.resize(
            images=x,
            size=(2 * height, 2 * width),
            method="nearest",
        )

        # TODO: clean up after PyTorch side is done
        if self.use_conv:

            # Pad with `1` element
            paddings = [(0, 0), (1, 1), (1, 1), (0, 0)]
            x = tf.pad(x, paddings, mode='CONSTANT', constant_values=0)

            if self._conv_name == "conv":
                x = self.conv(x)
            else:
                x = self.Conv2d_0(x)

        return x


class TFDownsample2D(tf.keras.layers.Layer):
    """
    A downsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs. :param use_conv: a bool determining if a convolution is
    applied. :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv=False, padding=1, out_channels=None, _conv_name="conv"):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.padding = padding
        stride = 2

        # TODO: clean up after PyTorch side is done
        if _conv_name == "Conv2d_0":
            _conv_name = "conv"

        if use_conv:
            conv = tf.keras.layers.Conv2D(self.out_channels, kernel_size=3, strides=stride, padding="valid", name=_conv_name)
        else:
            assert self.channels == self.out_channels
            conv = tf.keras.layers.AveragePooling2D(
                pool_size=stride,
                strides=stride,
                padding='valid',
                name=_conv_name,
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


class TFFirUpsample2D(tf.keras.layers.Layer):

    def __init__(self, channels=None, out_channels=None, use_conv=False, fir_kernel=(1, 3, 3, 1)):
        super().__init__()
        out_channels = out_channels if out_channels else channels
        if use_conv:
            # manual padding in self.call()
            self.Conv2d_0 = tf.keras.layers.Conv2D(filters=out_channels, kernel_size=3, strides=1, padding="VALID", name="Conv2d_0")
        self.use_conv = use_conv
        self.fir_kernel = fir_kernel
        self.out_channels = out_channels

    def _upsample_2d(self, x, weight=None, kernel=None, factor=2, gain=1):
        """Fused `upsample_2d()` followed by `Conv2d()`.

        Args:
        Padding is performed only once at the beginning, not between the operations. The fused op is considerably more
        efficient than performing the same calculation using standard TensorFlow ops. It supports gradients of arbitrary:
        order.
        x: Input tensor of the shape `[N, C, H, W]` or `[N, H, W,
            C]`.
        weight: Weight tensor of the shape `[filterH, filterW, inChannels,
            outChannels]`. Grouped convolution can be performed by `inChannels = x.shape[0] // numGroups`.
        kernel: FIR filter of the shape `[firH, firW]` or `[firN]`
            (separable). The default is `[1] * factor`, which corresponds to nearest-neighbor upsampling.
        factor: Integer upsampling factor (default: 2). gain: Scaling factor for signal magnitude (default: 1.0).

        The original TF implementation: https://github.com/NVlabs/stylegan2/blob/bf0fe0baba9fc7039eae0cac575c1778be1ce3e3/dnnlib/tflib/ops/upfirdn_2d.py#L234

        Returns:
        Tensor of the shape `[N, C, H * factor, W * factor]` or `[N, H * factor, W * factor, C]`, and same datatype as
        `x`.
        """

        assert isinstance(factor, int) and factor >= 1

        # Setup filter kernel.
        if kernel is None:
            kernel = [1] * factor

        # setup kernel
        kernel = tf.constant(kernel, dtype=tf.float32)
        if tf.rank(kernel) == 1:
            kernel = tf.tensordot(kernel, kernel, axes=0)
        kernel = kernel / tf.reduce_sum(kernel)

        kernel = kernel * (gain * (factor**2))

        if self.use_conv:
            # `weight` shape: [K_H, K_W, K_C_in, C_out]

            # `weight` could be `ResourceVariable`
            assert weight.shape.rank == 4

            convH = weight.shape[0]
            convW = weight.shape[1]
            assert convW == convH
            inC = weight.shape[2]
            outC = weight.shape[3]

            p = (kernel.shape[0] - factor) - (convW - 1)

            # (N, H, W, C)
            # Determine data dimensions.
            stride = [1, factor, factor, 1]
            output_shape = [x.shape[0], (x.shape[1] - 1) * factor + convH, (x.shape[2] - 1) * factor + convW, outC]
            num_groups = x.shape[3] // inC

            # Transpose weights.
            # `tf.nn.conv2d_transpose` requires `kernel` to have shape `(H, W, C_out, C_in]`.
            weight = tf.reshape(weight, [convH, convW, inC, num_groups, -1])
            weight = tf.transpose(weight[::-1, ::-1], [0, 1, 4, 3, 2])
            weight = tf.reshape(weight, [convH, convW, -1, num_groups * inC])

            x = tf.nn.conv2d_transpose(x, weight, output_shape=output_shape, strides=stride, padding='VALID', data_format="NHWC")

            x = upfirdn2d_native(x, kernel, pad=((p + 1) // 2 + factor - 1, p // 2 + 1))
        else:
            p = kernel.shape[0] - factor
            x = upfirdn2d_native(
                x, kernel, up=factor, pad=((p + 1) // 2 + factor - 1, p // 2)
            )

        return x

    def build(self, input_shape):

        if self.use_conv:
            sample = tf.random.normal(shape=input_shape, dtype=tf.float32)
            self.Conv2d_0(sample)

    def call(self, x):
        if self.use_conv:
            height = self._upsample_2d(x, self.Conv2d_0.kernel, kernel=self.fir_kernel)
            height = height + self.Conv2d_0.bias
        else:
            height = self._upsample_2d(x, kernel=self.fir_kernel, factor=2)

        return height


def upsample_2d(x, kernel=None, factor=2, gain=1):
    r"""Upsample2D a batch of 2D images with the given filter.

    Args:
    Accepts a batch of 2D images of the shape `[N, C, H, W]` or `[N, H, W, C]` and upsamples each image with the given
    filter. The filter is normalized so that if the input pixels are constant, they will be scaled by the specified
    `gain`. Pixels outside the image are assumed to be zero, and the filter is padded with zeros so that its shape is a:
    multiple of the upsampling factor.
        x: Input tensor of the shape `[N, C, H, W]` or `[N, H, W,
          C]`.
        k: FIR filter of the shape `[firH, firW]` or `[firN]`
          (separable). The default is `[1] * factor`, which corresponds to nearest-neighbor upsampling.
        factor: Integer upsampling factor (default: 2). gain: Scaling factor for signal magnitude (default: 1.0).

    The original TF implementation: https://github.com/NVlabs/stylegan2/blob/bf0fe0baba9fc7039eae0cac575c1778be1ce3e3/dnnlib/tflib/ops/upfirdn_2d.py#L202

    Returns:
        Tensor of the shape `[N, C, H * factor, W * factor]`
    """
    assert isinstance(factor, int) and factor >= 1
    if kernel is None:
        kernel = [1] * factor

    kernel = tf.constant(kernel, dtype=tf.float32)
    if tf.rank(kernel) == 1:
        kernel = tf.tensordot(kernel, kernel, axes=0)
    kernel = kernel / tf.reduce_sum(kernel)

    kernel = kernel * (gain * (factor**2))
    p = kernel.shape[0] - factor
    return upfirdn2d_native(
        x, kernel, up=factor, pad=((p + 1) // 2 + factor - 1, p // 2)
    )


def downsample_2d(x, kernel=None, factor=2, gain=1):
    r"""Downsample2D a batch of 2D images with the given filter.

    Args:
    Accepts a batch of 2D images of the shape `[N, C, H, W]` or `[N, H, W, C]` and downsamples each image with the
    given filter. The filter is normalized so that if the input pixels are constant, they will be scaled by the
    specified `gain`. Pixels outside the image are assumed to be zero, and the filter is padded with zeros so that its
    shape is a multiple of the downsampling factor.
        x: Input tensor of the shape `[N, C, H, W]` or `[N, H, W,
          C]`.
        kernel: FIR filter of the shape `[firH, firW]` or `[firN]`
          (separable). The default is `[1] * factor`, which corresponds to average pooling.
        factor: Integer downsampling factor (default: 2). gain: Scaling factor for signal magnitude (default: 1.0).

    Returns:
        Tensor of the shape `[N, C, H // factor, W // factor]`
    """

    assert isinstance(factor, int) and factor >= 1
    if kernel is None:
        kernel = [1] * factor

    kernel = tf.constant(kernel, dtype=tf.float32)
    if tf.rank(kernel) == 1:
        kernel = tf.tensordot(kernel, kernel, axes=0)
    kernel = kernel / tf.reduce_sum(kernel)

    kernel = kernel * gain
    p = kernel.shape[0] - factor
    return upfirdn2d_native(
        x, kernel, down=factor, pad=((p + 1) // 2, p // 2)
    )


# ======================================================================================================================
# Wrappers which have the corresponding entries in `resnet.py`


def upfirdn2d_native(input, kernel, up=1, down=1, pad=(0, 0)):

    return _upfirdn_2d_ref(
        input, kernel, upx=up, upy=up, downx=down, downy=down, padx0=pad[0], padx1=pad[1], pady0=pad[0], pady1=pad[1]
    )

# ======================================================================================================================
# Copied from https://github.com/NVlabs/stylegan2/blob/master/dnnlib/tflib/ops/upfirdn_2d.py with modifications.


# Copied from https://github.com/NVlabs/stylegan2/blob/bf0fe0baba9fc7039eae0cac575c1778be1ce3e3/dnnlib/tflib/ops/upfirdn_2d.py#L66
def _upfirdn_2d_ref(x, k, upx, upy, downx, downy, padx0, padx1, pady0, pady1):
    """Slow reference implementation of `upfirdn_2d()` using standard TensorFlow ops.

    Some modifications:
      - some numpy operation are (or will be) replaced by TF ops
      - remove `.value` from tensor shape, i.e. `x.shape[1].value -> x.shape[1]`

    TODO: (ydshieh) Understand FIR filtering.

    For argument meanings, check https://github.com/NVlabs/stylegan2/blob/bf0fe0baba9fc7039eae0cac575c1778be1ce3e3/dnnlib/tflib/ops/upfirdn_2d.py#L19
    """

    # Input `x` expect (N, H, W, C) format.
    x = tf.convert_to_tensor(x)
    k = tf.constant(k, dtype=tf.float32)
    assert x.shape.rank == 4
    inH = x.shape[1]
    inW = x.shape[2]
    minorDim = _shape(x, 3)
    kernelH, kernelW = k.shape
    assert inW >= 1 and inH >= 1
    assert kernelW >= 1 and kernelH >= 1
    assert isinstance(upx, int) and isinstance(upy, int)
    assert isinstance(downx, int) and isinstance(downy, int)
    assert isinstance(padx0, int) and isinstance(padx1, int)
    assert isinstance(pady0, int) and isinstance(pady1, int)

    # Upsample (insert zeros).
    x = tf.reshape(x, [-1, inH, 1, inW, 1, minorDim])
    x = tf.pad(x, [[0, 0], [0, 0], [0, upy - 1], [0, 0], [0, upx - 1], [0, 0]])
    x = tf.reshape(x, [-1, inH * upy, inW * upx, minorDim])

    # Pad (crop if negative).
    x = tf.pad(x, [[0, 0], [max(pady0, 0), max(pady1, 0)], [max(padx0, 0), max(padx1, 0)], [0, 0]])
    x = x[:, max(-pady0, 0) : x.shape[1] - max(-pady1, 0), max(-padx0, 0) : x.shape[2] - max(-padx1, 0), :]

    # Convolve with filter.
    # (N, C, H, W)
    x = tf.transpose(x, [0, 3, 1, 2])

    # --------------------------------------------------------------------------------
    # Original implementation uses (N * C, 1, H, W)
    # x = tf.reshape(x, [-1, 1, inH * upy + pady0 + pady1, inW * upx + padx0 + padx1])

    # Here we use (N * C, H, W, 1) so it could run on CPU with `tf.nn.conv2d`
    x = tf.reshape(x, [-1, inH * upy + pady0 + pady1, inW * upx + padx0 + padx1, 1])
    # --------------------------------------------------------------------------------

    w = tf.constant(k[::-1, ::-1, tf.newaxis, tf.newaxis], dtype=x.dtype)

    # Original implementation uses `NCHW` here.
    x = tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='VALID', data_format='NHWC')

    x = tf.reshape(x, [-1, minorDim, inH * upy + pady0 + pady1 - kernelH + 1, inW * upx + padx0 + padx1 - kernelW + 1])
    x = tf.transpose(x, [0, 2, 3, 1])

    # Downsample (throw away pixels).
    return x[:, ::downy, ::downx, :]


# Copied from https://github.com/NVlabs/stylegan2/blob/bf0fe0baba9fc7039eae0cac575c1778be1ce3e3/dnnlib/tflib/ops/upfirdn_2d.py#L337
def _shape(tf_expr, dim_idx):
    if tf_expr.shape.rank is not None:
        dim = tf_expr.shape[dim_idx]
        if dim is not None:
            return dim
    return tf.shape(tf_expr)[dim_idx]
