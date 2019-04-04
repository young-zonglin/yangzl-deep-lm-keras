import tensorflow as tf
from keras import backend as K
from keras.engine.topology import Layer
from keras.initializers import Ones, Zeros


class LayerNormalization(Layer):
    def __init__(self, eps=1e-6, **kwargs):
        super(LayerNormalization, self).__init__(**kwargs)
        self.eps = eps
        self.gamma = None
        self.beta = None

    def build(self, input_shape):
        # Create trainable weight variables for this layer.
        # 不同的行共享gamma和beta
        self.gamma = self.add_weight(name='gamma', shape=input_shape[-1:],
                                     initializer=Ones(), trainable=True)
        self.beta = self.add_weight(name='beta', shape=input_shape[-1:],
                                    initializer=Zeros(), trainable=True)
        super(LayerNormalization, self).build(input_shape)  # Be sure to call this at the end

    def call(self, inputs, **kwargs):
        mean = K.mean(inputs, axis=-1, keepdims=True)
        std = K.std(inputs, axis=-1, keepdims=True)
        # 类似于BN，LN在对样本归一化后也有缩放和平移操作
        # Python中的广播
        return self.gamma * (inputs - mean) / (std + self.eps) + self.beta

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {'eps': self.eps}
        base_config = super(LayerNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class AvgEmb(Layer):
    def __init__(self, word_vec_dim, **kwargs):
        super(AvgEmb, self).__init__(**kwargs)
        self.word_vec_dim = word_vec_dim

    # 模板方法模式
    def call(self, inputs, **kwargs):
        inputs = tf.reduce_mean(inputs, axis=1, keepdims=True)
        # return Reshape([self.word_vec_dim])(X)
        return tf.reshape(inputs, [-1, self.word_vec_dim])

    # 由于是静态计算图的框架，shape都并不可靠，可能不是预期的值
    # 尽量使用已知值
    def compute_output_shape(self, input_shape, **kwargs):
        return input_shape[0], self.word_vec_dim

    # 和保存相关的方法
    # config = layer.get_config() or model.get_config() => 包含这个层配置信息的dict
    # layer = Layer.from_config(config) or
    # model = Model.from_config(config) or Sequential.from_config(config)
    # 由于Keras其他层没有重写from_config方法，我的自定义层也不重写
    def get_config(self):
        config = {'word_vec_dim': self.word_vec_dim}
        base_config = super(AvgEmb, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

