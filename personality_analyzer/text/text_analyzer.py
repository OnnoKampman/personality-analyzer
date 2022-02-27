import keras
from keras.models import Model
from keras.layers import Input, Dense, Conv1D, Dropout, Activation, Reshape, concatenate, Add
from keras.layers import GlobalMaxPooling1D, GlobalAveragePooling1D
from keras import regularizers
import logging
import tensorflow as tf

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


class TextPersonalityAnalyzer:
    """
    Main class to predict Big Five personality from language/text.
    """
    def __init__(
            self, embed_dim: int, depth,
            patch_size1: int, patch_size2: int, patch_size3: int,
            stride, hidden_units, labels_dim, L2_beta,
            kernel_initializer, bias_initializer,
            dropout_prob: float = 0.6
    ):
        inputs = Input(shape=(None, embed_dim))

        c1 = Conv1D(
            filters=depth, kernel_size=patch_size1,
            strides=stride, padding='same',
            use_bias=True,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            activation='relu'
        )(inputs)
        c2 = Conv1D(
            filters=depth, kernel_size=patch_size2,
            strides=stride, padding='same',
            use_bias=True,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            activation='relu'
        )(inputs)
        c3 = Conv1D(
            filters=depth, kernel_size=patch_size3,
            strides=stride, padding='same',
            use_bias=True,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            activation='relu'
        )(inputs)

        a1 = GlobalMaxPooling1D()(c1)
        a2 = GlobalMaxPooling1D()(c2)
        a3 = GlobalMaxPooling1D()(c3)

        m = keras.layers.concatenate([a1, a2, a3], axis=1)

        dr = Dropout(dropout_prob)(m)

        d1 = Dense(
            units=hidden_units, use_bias=True,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=regularizers.l2(L2_beta),
            activation='relu'
        )(dr)
        pr = Dense(
            units=labels_dim, use_bias=True,
            kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
            kernel_regularizer=regularizers.l2(L2_beta),
            activation='sigmoid'
        )(d1)

        self.model = Model(inputs=inputs, outputs=pr)

    def train(self, loss_function, optimizer):
        """
        Train routine.
        :param loss_function: str; such as 'MSE'.
        :param optimizer:
        :return:
        """
        self.model.compile(
            loss=loss_function,
            # metrics=['accuracy'],
            optimizer=optimizer
        )
        self.model.summary()

    def predict(self):

        return self.model
