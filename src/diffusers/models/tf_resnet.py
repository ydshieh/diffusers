from functools import partial
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


class TFUpsample2D(tf.keras.layers.Layer):
    """
    An upsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs. :param use_conv: a bool determining if a convolution is
    applied. :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 upsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv=False, use_conv_transpose=False, out_channels=None, _conv_name="conv", **kwargs):
        super().__init__(**kwargs)
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
        assert x.shape[-1] == self.channels
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

    def __init__(self, channels, use_conv=False, padding=1, out_channels=None, _conv_name="conv", **kwargs):
        super().__init__(**kwargs)
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.padding = padding
        stride = 2

        # TODO: clean up after PyTorch side is done
        if _conv_name == "Conv2d_0":
            _conv_name = "conv"

        if use_conv:
            # manual padding in self.call()
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

    def __init__(self, channels=None, out_channels=None, use_conv=False, fir_kernel=(1, 3, 3, 1), **kwargs):
        super().__init__(**kwargs)
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


class TFFirDownsample2D(tf.keras.layers.Layer):

    def __init__(self, channels=None, out_channels=None, use_conv=False, fir_kernel=(1, 3, 3, 1), **kwargs):
        super().__init__(**kwargs)
        out_channels = out_channels if out_channels else channels
        if use_conv:
            # manual padding in self.call()
            self.Conv2d_0 = tf.keras.layers.Conv2D(filters=out_channels, kernel_size=3, strides=1, padding="VALID", name="Conv2d_0")
        self.use_conv = use_conv
        self.fir_kernel = fir_kernel
        self.out_channels = out_channels

    def _downsample_2d(self, x, weight=None, kernel=None, factor=2, gain=1):
        """Fused `Conv2d()` followed by `downsample_2d()`.

        Args:
        Padding is performed only once at the beginning, not between the operations. The fused op is considerably more
        efficient than performing the same calculation using standard TensorFlow ops. It supports gradients of arbitrary:
        order.
            x: Input tensor of the shape `[N, C, H, W]` or `[N, H, W, C]`. w: Weight tensor of the shape `[filterH,
            filterW, inChannels, outChannels]`. Grouped convolution can be performed by `inChannels = x.shape[0] //
            numGroups`. k: FIR filter of the shape `[firH, firW]` or `[firN]` (separable). The default is `[1] *
            factor`, which corresponds to average pooling. factor: Integer downsampling factor (default: 2). gain:
            Scaling factor for signal magnitude (default: 1.0).

        The original TF implementation: https://github.com/NVlabs/stylegan2/blob/bf0fe0baba9fc7039eae0cac575c1778be1ce3e3/dnnlib/tflib/ops/upfirdn_2d.py#L296

        Returns:
            Tensor of the shape `[N, C, H // factor, W // factor]` or `[N, H // factor, W // factor, C]`, and same
            datatype as `x`.
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

        kernel = kernel * gain

        if self.use_conv:
            # `weight` shape: [K_H, K_W, K_C_in, C_out]

            # `weight` could be `ResourceVariable`
            assert weight.shape.rank == 4

            convH = weight.shape[0]
            convW = weight.shape[1]
            assert convW == convH

            p = (kernel.shape[0] - factor) + (convW - 1)

            # (N, H, W, C)
            # Determine data dimensions.
            stride = [1, factor, factor, 1]

            x = upfirdn2d_native(x, kernel, pad=((p + 1) // 2, p // 2))
            x = tf.nn.conv2d(x, weight, strides=stride, padding='VALID', data_format="NHWC")
        else:
            p = kernel.shape[0] - factor
            x = upfirdn2d_native(
                x, kernel, down=factor, pad=((p + 1) // 2, p // 2)
            )

        return x

    def build(self, input_shape):

        if self.use_conv:
            sample = tf.random.normal(shape=input_shape, dtype=tf.float32)
            self.Conv2d_0(sample)

    def call(self, x):
        if self.use_conv:
            x = self._downsample_2d(x, self.Conv2d_0.kernel, kernel=self.fir_kernel)
            x = x + self.Conv2d_0.bias
        else:
            x = self._downsample_2d(x, kernel=self.fir_kernel, factor=2)

        return x


class TFResnetBlock2D(tf.keras.layers.Layer):
    def __init__(
        self,
        *,
        in_channels,
        out_channels=None,
        conv_shortcut=False,
        dropout=0.0,
        temb_channels=512,
        groups=32,
        groups_out=None,
        pre_norm=True,
        eps=1e-6,
        non_linearity="swish",
        time_embedding_norm="default",
        kernel=None,
        output_scale_factor=1.0,
        use_in_shortcut=None,
        up=False,
        down=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.pre_norm = pre_norm
        self.pre_norm = True
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut
        self.time_embedding_norm = time_embedding_norm
        self.kernel = kernel
        self.up = up
        self.down = down
        self.output_scale_factor = output_scale_factor

        if groups_out is None:
            groups_out = groups

        self.norm1 = TFGroupNormalization(groups=groups, epsilon=eps, center=True, scale=True, name="norm1")

        self.conv1 = tf.keras.layers.Conv2D(filters=out_channels, kernel_size=3, strides=1, padding="VALID", name="conv1")

        if temb_channels is not None:
            self.time_emb_proj = tf.keras.layers.Dense(units=out_channels, name="time_emb_proj")
        else:
            self.time_emb_proj = None

        self.norm2 = TFGroupNormalization(groups=groups_out, epsilon=eps, center=True, scale=True, name="norm2")
        self.dropout = tf.keras.layers.Dropout(dropout, name="dropout")
        self.conv2 = tf.keras.layers.Conv2D(filters=out_channels, kernel_size=3, strides=1, padding="VALID", name="conv2")

        # TODO: (Question) What's the difference between "swish" and "silu"?
        # Looks like the implementations in `resnet.py` are identical.
        if non_linearity == "swish":
            self.nonlinearity = tf.keras.activations.swish
        elif non_linearity == "mish":
            self.nonlinearity = Mish()
        elif non_linearity == "silu":
            self.nonlinearity = tf.keras.activations.swish

        self.upsample = self.downsample = None
        if self.up:
            if kernel == "fir":
                fir_kernel = (1, 3, 3, 1)
                self.upsample = lambda x: upsample_2d(x, kernel=fir_kernel)
            elif kernel == "sde_vp":
                self.upsample = partial(tf.image.resize, method="nearest")
            else:
                self.upsample = TFUpsample2D(in_channels, use_conv=False, name="upsample")
        elif self.down:
            if kernel == "fir":
                fir_kernel = (1, 3, 3, 1)
                self.downsample = lambda x: downsample_2d(x, kernel=fir_kernel)
            elif kernel == "sde_vp":
                self.downsample = partial(tf.nn.avg_pool2d, ksize=2, strides=2, padding="VALID")
            else:
                self.downsample = TFDownsample2D(in_channels, use_conv=False, padding=1, name="op")

        self.use_in_shortcut = self.in_channels != self.out_channels if use_in_shortcut is None else use_in_shortcut

        self.conv_shortcut = None
        if self.use_in_shortcut:
            self.conv_shortcut = tf.keras.layers.Conv2D(filters=out_channels, kernel_size=1, strides=1, padding="VALID", name="conv_shortcut")

    def call(self, x, temb=None):
        hidden_states = x

        # make sure hidden states is in float32
        # when running in half-precision
        hidden_states = self.norm1(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)

        if self.upsample is not None:
            if self.kernel == "sde_vp":
                (height, width) = tf.shape(x)[1:3]
                x = self.upsample(x, size=(2 * height, 2 * width))
                (height, width) = tf.shape(hidden_states)[1:3]
                hidden_states = self.upsample(hidden_states, size=(2 * height, 2 * width))
            else:
                x = self.upsample(x)
                hidden_states = self.upsample(hidden_states)
        elif self.downsample is not None:
            x = self.downsample(x)
            hidden_states = self.downsample(hidden_states)

        # Pad with `1` element
        paddings = [(0, 0), (1, 1), (1, 1), (0, 0)]
        hidden_states = tf.pad(hidden_states, paddings, mode='CONSTANT', constant_values=0)
        hidden_states = self.conv1(hidden_states)

        if temb is not None:
            temb = self.time_emb_proj(self.nonlinearity(temb))[:, :, tf.newaxis, tf.newaxis]
            hidden_states = hidden_states + temb

        # make sure hidden states is in float32
        # when running in half-precision
        hidden_states = self.norm2(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)

        hidden_states = self.dropout(hidden_states)
        # Pad with `1` element
        paddings = [(0, 0), (1, 1), (1, 1), (0, 0)]
        hidden_states = tf.pad(hidden_states, paddings, mode='CONSTANT', constant_values=0)
        hidden_states = self.conv2(hidden_states)

        if self.conv_shortcut is not None:
            x = self.conv_shortcut(x)

        out = (x + hidden_states) / self.output_scale_factor

        return out


class Mish(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, x):
        return x * tf.math.tanh(tf.math.softplus(x))


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
