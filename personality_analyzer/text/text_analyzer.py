import keras
from keras.models import Model
from keras.layers import Input, Dense, Conv1D, Dropout, Activation, Reshape, concatenate, Merge, Add
from keras.layers import GlobalMaxPooling1D, GlobalAveragePooling1D
from keras import regularizers
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


class TextPersonalityAnalyzer:

    def __init__(self, embed_dim, depth,
                 patch_size1, patch_size2, patch_size3,
                 stride, hidden_units, labels_dim, L2_beta,
                 kernel_initializer, bias_initializer):

        inputs = Input(shape=(None, embed_dim))

        c1 = Conv1D(filters=depth, kernel_size=patch_size1,
                    strides=stride, padding='same',
                    use_bias=True, bias_initializer=bias_initializer,
                    kernel_initializer=kernel_initializer,
                    activation='relu')(inputs)
        c2 = Conv1D(filters=depth, kernel_size=patch_size2,
                    strides=stride, padding='same',
                    use_bias=True, kernel_initializer=kernel_initializer,
                    bias_initializer=bias_initializer,
                    activation='relu')(inputs)
        c3 = Conv1D(filters=depth, kernel_size=patch_size3,
                    strides=stride, padding='same',
                    use_bias=True, kernel_initializer=kernel_initializer,
                    bias_initializer=bias_initializer,
                    activation='relu')(inputs)

        a1 = GlobalMaxPooling1D()(c1)
        a2 = GlobalMaxPooling1D()(c2)
        a3 = GlobalMaxPooling1D()(c3)

        m = keras.layers.concatenate([a1, a2, a3], axis=1)

        dr = Dropout(0.6)(m)

        d1 = Dense(units=hidden_units, use_bias=True,
                   kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
                   kernel_regularizer=regularizers.l2(L2_beta), activation='relu')(dr)
        pr = Dense(units=labels_dim, use_bias=True,
                   kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
                   kernel_regularizer=regularizers.l2(L2_beta), activation='sigmoid')(d1)

        self.model = Model(inputs=inputs, outputs=pr)

    def train(self, loss_function, optimizer):
        """
        Train routine.
        :param loss_function: str; such as 'MSE'.
        :param optimizer:
        :return:
        """
        self.model.compile(loss=loss_function,
                           # metrics=['accuracy'],
                           optimizer=optimizer)
        self.model.summary()

    def predict(self):

        return self.model
