import numpy as np
from keras import layers, initializers
from keras.layers.core import Layer
import keras.backend as K


class GradientReversal(Layer):
    '''Flip the sign of gradient during training.'''
    def __init__(self, hp_lambda, **kwargs):
        super(GradientReversal, self).__init__(**kwargs)
        self.supports_masking = False
        self.hp_lambda = hp_lambda

    def build(self, input_shape):
        self.trainable_weights = []

    def call(self, x, mask=None):
        if K.backend() == 'theano':
            import theano

            class ReverseGradient(theano.Op):
                """ theano operation to reverse the gradients
                Introduced in http://arxiv.org/pdf/1409.7495.pdf
                """
                view_map = {0: [0]}

                __props__ = ('hp_lambda',)

                def __init__(self, hp_lambda):
                    super(ReverseGradient, self).__init__()
                    self.hp_lambda = hp_lambda

                def make_node(self, x):
                    assert hasattr(self, '_props'), "Your version of theano is too old to support __props__."
                    x = theano.tensor.as_tensor_variable(x)
                    return theano.Apply(self, [x], [x.type()])

                def perform(self, node, inputs, output_storage, params=None):
                    xin, = inputs
                    xout, = output_storage
                    xout[0] = xin

                def grad(self, input, output_gradients):
                    return [-self.hp_lambda * output_gradients[0]]

                def infer_shape(self, node, i0_shapes):
                    return i0_shapes

            return ReverseGradient(self.hp_lambda)(x)
        elif K.backend() == 'tensorflow':
            import tensorflow as tf

            def reverse_gradient(X, hp_lambda):
                # Flips the sign of the incoming gradient during training.
                try:
                    reverse_gradient.num_calls += 1
                except AttributeError:
                    reverse_gradient.num_calls = 1

                grad_name = "GradientReversal%d" % reverse_gradient.num_calls

                @tf.RegisterGradient(grad_name)
                def _flip_gradients(op, grad):
                    return [tf.negative(grad) * hp_lambda]

                g = K.get_session().graph
                with g.gradient_override_map({'Identity': grad_name}):
                    y = tf.identity(X)

                return y
            return reverse_gradient(x, self.hp_lambda)

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_output_shape_for(self, input_shape):
        return input_shape

    def get_config(self):
        config = {"name": self.__class__.__name__,
                  "lambda": self.hp_lambda}
        base_config = super(GradientReversal, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Affine3D(Layer):
    """Spatial Transformer Layer
    Implements a spatial transformer layer as described in [1]_.
    Borrowed from [2]_:
    downsample_fator : float
        A value of 1 will keep the orignal size of the image.
        Values larger than 1 will down sample the image. Values below 1 will
        upsample the image.
        example image: height= 100, width = 200
        downsample_factor = 2
        output image will then be 50, 100
    References
    ----------
    .. [1]  Spatial Transformer Networks
            Max Jaderberg, Karen Simonyan, Andrew Zisserman, Koray Kavukcuoglu
            Submitted on 5 Jun 2015
    .. [2]  https://github.com/skaae/transformer_network/blob/master/transformerlayer.py
    .. [3]  https://github.com/EderSantana/seya/blob/keras1/seya/layers/attention.py
    """

    def __init__(self,
                 **kwargs):
        super(Affine3D, self).__init__(**kwargs)

    def build(self, input_shape):
        self.affine = self.add_weight(name='affine',
                                      shape=(1, 3, 4),
                                      initializer='uniform',
                                      trainable=True)

    def call(self, input_layer, mask=None):
        output = self._transform(self.affine, input_layer)
        return output

    @staticmethod
    def _linspace(start, stop, num):
        # Theano linspace. Behaves similar to np.linspace
        start = K.cast(start, K.floatx)
        stop = K.cast(stop, K.floatx)
        num = K.cast(num, K.floatx)
        step = (stop - start) / (num - 1)
        return K.arange(num, dtype=K.floatx) * step + start

    @staticmethod
    def _meshgrid(height, width, depth):
        # This function is the grid generator from eq. (1) in reference [1].
        # It is equivalent to the following numpy code:
        #  x_t, y_t,z_t = np.meshgrid(np.linspace(-1, 1, width),
        #                         np.linspace(-1, 1, height))
        #  ones = np.ones(np.prod(x_t.shape))
        #  grid = np.vstack([x_t.flatten(), y_t.flatten(), ones])
        # It is implemented in Theano instead to support symbolic grid sizes.
        # Note: If the image size is known at layer construction time, we could
        # compute the meshgrid offline in numpy instead of doing it dynamically
        # in Theano. However, it hardly affected performance when we tried.
        x_t = K.dot(
            K.reshape(K.dot(
                Affine3D._linspace(-1.0, 1.0, height).dimshuffle(0, 'x'),
                K.ones((1, width))), (height, width, 1)),
            K.ones((1, 1, depth))
        )
        y_t = K.dot(
            K.reshape(K.dot(
                K.ones((height, 1)),
                Affine3D._linspace(-1.0, 1.0, width).dimshuffle('x', 0)), (height, width, 1)),
            K.ones((1, 1, depth))
        )
        z_t = K.dot(K.ones((height, width, 1)), K.reshape(Affine3D._linspace(-1.0, 1.0, depth), (1, 1, -1)))

        x_t_flat = x_t.reshape((1, -1))
        y_t_flat = y_t.reshape((1, -1))
        z_t_flat = z_t.reshape((1, -1))
        ones = K.ones_like(x_t_flat)
        grid = K.concatenate([x_t_flat, y_t_flat, z_t_flat, ones], axis=0)
        return grid

    @staticmethod
    def _interpolate(im, x, y, z, out_height, out_width, out_depth):
        # *_f are floats
        num_batch, height, width, depth, channels = im.shape
        height_f = K.cast(height, K.floatx)
        width_f = K.cast(width, K.floatx)
        depth_f = K.cast(depth, K.floatx)

        # clip coordinates to [-1, 1]
        x = K.clip(x, -1, 1)
        y = K.clip(y, -1, 1)
        z = K.clip(z, -1, 1)

        # scale coordinates from [-1, 1] to [0, width/height/depth - 1]
        x = (x + 1) / 2 * (width_f - 1)
        y = (y + 1) / 2 * (height_f - 1)
        z = (z + 1) / 2 * (depth_f - 1)

        # obtain indices of the 2x2x2 pixel neighborhood surrounding the coordinates;
        # we need those in floatX for interpolation and in int64 for indexing. for
        # indexing, we need to take care they do not extend past the image.
        x0_f = K.floor(x)
        y0_f = K.floor(y)
        z0_f = K.floor(z)
        x1_f = x0_f + 1
        y1_f = y0_f + 1
        z1_f = z0_f + 1
        x0 = K.cast(x0_f, 'int64')
        y0 = K.cast(y0_f, 'int64')
        z0 = K.cast(z0_f, 'int64')
        x1 = K.cast(K.minimum(x1_f, width_f - 1), 'int64')
        y1 = K.cast(K.minimum(y1_f, height_f - 1), 'int64')
        z1 = K.cast(K.minimum(z1_f, depth_f - 1), 'int64')

        # The input is [num_batch, height, width, depth, channels]. We do the lookup in
        # the flattened input, i.e [num_batch*height*width*depth, channels]. We need
        # to offset all indices to match the flat version
        dim1 = height * width * depth
        dim2 = width * depth
        dim3 = depth
        base = K.repeat(
            K.arange(num_batch, dtype='int64') * dim1, out_height * out_width * out_depth)
        base_y0 = base + y0 * dim2
        base_y1 = base + y1 * dim2
        base_x0 = x0 * dim3
        base_x1 = x1 * dim3
        idx_a = base_y0 + base_x0 + z0
        idx_b = base_y1 + base_x0 + z0
        idx_c = base_y0 + base_x1 + z0
        idx_d = base_y1 + base_x1 + z0
        idx_e = base_y0 + base_x0 + z1
        idx_f = base_y1 + base_x0 + z1
        idx_g = base_y0 + base_x1 + z1
        idx_h = base_y1 + base_x1 + z1

        # use indices to lookup pixels for all samples
        im_flat = im.reshape((-1, channels))
        Ia = im_flat[idx_a]
        Ib = im_flat[idx_b]
        Ic = im_flat[idx_c]
        Id = im_flat[idx_d]
        Ie = im_flat[idx_e]
        If = im_flat[idx_f]
        Ig = im_flat[idx_g]
        Ih = im_flat[idx_h]

        # calculate interpolated values
        wa = ((x1_f - x) * (y1_f - y) * (z1_f - z)).dimshuffle(0, 'x')
        wb = ((x1_f - x) * (y - y0_f) * (z1_f - z)).dimshuffle(0, 'x')
        wc = ((x - x0_f) * (y1_f - y) * (z1_f - z)).dimshuffle(0, 'x')
        wd = ((x - x0_f) * (y - y0_f) * (z1_f - z)).dimshuffle(0, 'x')
        we = ((x1_f - x) * (y1_f - y) * (z0_f - z)).dimshuffle(0, 'x')
        wf = ((x1_f - x) * (y - y0_f) * (z0_f - z)).dimshuffle(0, 'x')
        wg = ((x - x0_f) * (y1_f - y) * (z0_f - z)).dimshuffle(0, 'x')
        wh = ((x - x0_f) * (y - y0_f) * (z0_f - z)).dimshuffle(0, 'x')
        output = K.sum([wa * Ia, wb * Ib, wc * Ic, wd * Id, we * Ie, wf * If, wg * Ig, wh * Ih], axis=0)
        return output

    def _transform(self, theta, input_layer):
        num_batch, num_channels, height, width, depth = input_layer.shape
        theta = K.reshape(theta, (-1, 3, 4))

        # grid of (x_t, y_t, z_t, 1), eq (1) in ref [1]
        out_height = K.cast(height, 'int64')
        out_width = K.cast(width, 'int64')
        out_depth = K.cast(depth, 'int64')
        grid = Affine3D._meshgrid(out_height, out_width, out_depth)

        # Transform A x (x_t, y_t, z_t, 1)^T -> (x_s, y_s, z_s)
        T_g = K.dot(theta, grid)
        x_s = T_g[:, 0]
        y_s = T_g[:, 1]
        z_s = T_g[:, 2]
        x_s_flat = x_s.flatten()
        y_s_flat = y_s.flatten()
        z_s_flat = z_s.flatten()

        # dimshuffle input to  (bs, height, width, depth, channels)
        input_dim = input_layer.dimshuffle(0, 2, 3, 4, 1)
        input_transformed = Affine3D._interpolate(
            input_dim, x_s_flat, y_s_flat, z_s_flat,
            out_height, out_width, out_depth)

        output = K.reshape(
            input_transformed, (num_batch, out_height, out_width, out_depth, num_channels))
        output = output.dimshuffle(0, 4, 1, 2, 3)  # dimshuffle to conv format
        return output


# CAPSULE STUFF
class Length(layers.Layer):
    """
    Compute the length of vectors. This is used to compute a Tensor that has the same shape with y_true in margin_loss
    inputs: shape=[dim_1, ..., dim_{n-1}, dim_n]
    output: shape=[dim_1, ..., dim_{n-1}]
    """

    def __call__(self, inputs, **kwargs):
        return K.sqrt(K.sum(K.square(inputs), -1))

    def compute_output_shape(self, input_shape):
        return input_shape[:-1]


class Mask(layers.Layer):
    """
    Mask a Tensor with shape=[None, d1, d2] by the max value in axis=1.
    Output shape: [None, d2]
    """

    def __call__(self, inputs, **kwargs):
        # use true label to select target capsule, shape=[batch_size, num_capsule]
        if type(inputs) is list:  # true label is provided with shape = [batch_size, n_classes], i.e. one-hot code.
            assert len(inputs) == 2
            inputs, mask = inputs
            mask = K.expand_dims(mask, -1)
        else:  # if no true label, mask by the max length of vectors of capsules. Used for prediction
            x = K.sqrt(K.sum(K.square(inputs), -1, True))
            # Enlarge the range of values in x to make max(new_x)=1 and others < 0
            x = (x - K.max(x, 1, True)) / K.epsilon() + 1
            mask = K.clip(x, 0, 1)  # the max value in x clipped to 1 and other to 0

        return K.batch_flatten(inputs * mask)  # masked inputs, shape = [None, num_capsule * dim_vector]

    def compute_output_shape(self, input_shape):
        if type(input_shape[0]) is tuple:  # true label provided
            return tuple([None, input_shape[0][1] * input_shape[0][2]])
        else:
            return tuple([None, input_shape[1] * input_shape[2]])


def squash(vectors, axis=-1):
    """
    The non-linear activation used in Capsule. It drives the length of a large vector to near 1 and small vector to 0
    :param vectors: some vectors to be squashed, N-dim tensor
    :param axis: the axis to squash
    :return: a Tensor with same shape as input vectors
    """
    s_squared_norm = K.sum(K.square(vectors), axis, keepdims=True)
    scale = s_squared_norm / (1 + s_squared_norm) / K.sqrt(s_squared_norm + K.epsilon())
    return scale * vectors


class CapsuleLayer(layers.Layer):
    """
    The capsule layer. It is similar to Dense layer. Dense layer has `in_num` inputs, each is a scalar, the output of the
    neuron from the former layer, and it has `out_num` output neurons. CapsuleLayer just expand the output of the neuron
    from scalar to vector. So its input shape = [None, input_num_capsule, input_dim_vector] and output shape = \
    [None, num_capsule, dim_vector]. For Dense Layer, input_dim_vector = dim_vector = 1.

    :param num_capsule: number of capsules in this layer
    :param dim_vector: dimension of the output vectors of the capsules in this layer
    :param num_routings: number of iterations for the routing algorithm
    """

    def __init__(self, num_capsule, dim_vector, num_routing=3,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 **kwargs):
        super(CapsuleLayer, self).__init__(**kwargs)
        self.num_capsule = num_capsule
        self.dim_vector = dim_vector
        self.num_routing = num_routing
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.input_num_capsule = 0
        self.input_dim_vector = 0
        self.W = None
        self.bias = None
        self.built = False

    def build(self, input_shape):
        assert len(input_shape) >= 3, "The input Tensor should have shape=[None, input_num_capsule, input_dim_vector]"
        self.input_num_capsule = input_shape[1]
        self.input_dim_vector = input_shape[2]

        # Transform matrix
        self.W = self.add_weight(
            shape=tuple([self.input_num_capsule, self.num_capsule, self.input_dim_vector, self.dim_vector]),
            initializer=self.kernel_initializer,
            name='W')

        # Coupling coefficient. The redundant dimensions are just to facilitate subsequent matrix calculation.
        self.bias = self.add_weight(shape=tuple([1, self.input_num_capsule, self.num_capsule, 1, 1]),
                                    initializer=self.bias_initializer,
                                    name='bias',
                                    trainable=False)
        self.built = True

    def call(self, inputs, **kwargs):
        # inputs.shape=[None, input_num_capsule, input_dim_vector]
        # Expand dims to [None, input_num_capsule, 1, 1, input_dim_vector]
        inputs_expand = K.expand_dims(K.expand_dims(inputs, 2), 2)

        # Replicate num_capsule dimension to prepare being multiplied by W
        # Now it has shape = [None, input_num_capsule, num_capsule, 1, input_dim_vector]
        inputs_tiled = K.tile(inputs_expand, [1, 1, self.num_capsule, 1, 1])

        # Begin: inputs_hat computation V1 ---------------------------------------------------------------------#
        # Compute `inputs * W` by expanding the first dim of W. More time-consuming and need batch_size.
        w_tiled = K.tile(K.expand_dims(self.W, 0), [-1, 1, 1, 1, 1])
        # Transformed vectors, inputs_hat.shape = [None, input_num_capsule, num_capsule, 1, dim_vector]
        inputs_hat = K.batch_dot(inputs_tiled, w_tiled, [4, 3])
        # End: inputs_hat computation V1 ---------------------------------------------------------------------#

        # Begin: routing algorithm V2, static -----------------------------------------------------------#
        # Routing algorithm V2. Use iteration. V2 and V1 both work without much difference on performance
        assert self.num_routing > 0, 'The num_routing should be > 0.'
        outputs = None
        for i in range(self.num_routing):
            permuted_bias = K.permute_dimensions(self.bias, (0, 1, 4, 3, 2))
            bias_shape = K.int_shape(permuted_bias)
            soft_bias = K.reshape(K.softmax(
                K.reshape(permuted_bias, (np.prod(bias_shape[:-1]), bias_shape[-1]))
            ), bias_shape)
            print(K.int_shape((soft_bias)), bias_shape)
            c = K.permute_dimensions(soft_bias, (0, 1, 4, 3, 2))
            # c = K.permute_dimensions(
            #     K.softmax(K.permute_dimensions(self.bias, (0, 1, 4, 3, 2))),
            #     (0, 1, 4, 3, 2)
            # )
            # c = tf.nn.softmax(self.bias, dim=2)  # dim=2 is the num_capsule dimension
            outputs = squash(K.sum(c * inputs_hat, 1, keepdims=True))

            # last iteration needs not compute bias which will not be passed to the graph any more anyway.
            if i != self.num_routing - 1:
                self.bias += K.sum(inputs_hat * outputs, -1, keepdims=True)
        # End: routing algorithm V2, static ------------------------------------------------------------#

        return K.reshape(outputs, [-1, self.num_capsule, self.dim_vector])

    def compute_output_shape(self, **kwargs):
        return tuple([None, self.num_capsule, self.dim_vector])


class PrimaryCap3D(layers.Conv3D):
    """
        Apply Conv3D `n_channels` times and concatenate all capsules
        :param inputs: 4D tensor, shape=[None, width, height, channels]
        :param dim_vector: the dim of the output vector of capsule
        :param n_channels: the number of types of capsules
        :return: output tensor, shape=[None, num_capsule, dim_vector]
    """

    def __init__(self, dim_vector, **kwargs):
        super(PrimaryCap3D, self).__init__(**kwargs)
        self.dim_vector = dim_vector

    def call(self, inputs, **kwargs):
        output = super(PrimaryCap3D, self).call(inputs)
        outputs = K.reshape(output, shape=[-1, self.dim_vector])
        return squash(outputs)

    def compute_output_shape(self, input_shape):
        vector_len = np.prod(super(PrimaryCap3D, self).compute_output_shape(input_shape)[1:]) / self.dim_vector
        assert vector_len > 0, "The layer has a 0 dimension"
        return tuple([None, vector_len, self.dim_vector])
