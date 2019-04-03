from keras import backend as K
from keras.engine.topology import Layer


def dot_product(x, kernel):
    if K.backend() == 'tensorflow':
        return K.squeeze(K.dot(x, K.expand_dims(kernel)), axis=-1)
    else:
        return K.dot(x, kernel)


class MeanOverTime(Layer):
    def __init__(self, **kwargs):
        self.supports_masking = True
        super(MeanOverTime, self).__init__(**kwargs)

    def call(self, x, mask=None):
        if mask is not None:
            mask = K.cast(mask, 'float32')
            if not K.any(mask):
                return K.mean(x, axis=1)
            else:
                return K.cast(x.sum(axis=1) / mask.sum(axis=1, keepdims=True),
                              K.floatx())
        else:
            return K.mean(x, axis=1)

    def get_output_shape_for(self, input_shape):
        return input_shape[0], input_shape[-1]

    def compute_mask(self, input, input_mask=None):
        return None





