import tensorflow as tf

from personality_analyzer.text.text_analyzer import TextPersonalityAnalyzer


if __name__ == "__main__":

    bias_initializer = tf.keras.initializers.Constant(value=0.05)
    kernel_initializer = tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.05, seed=None)
    adam = tf.keras.optimizers.Adam(
        learning_rate=0.0001,
        beta_1=0.8,
        beta_2=0.999,
        epsilon=1e-08,
        decay=0.00001
    )
    model = TextPersonalityAnalyzer(
        embed_dim=8,
        depth=16,
        patch_size1=3,
        patch_size2=4,
        patch_size3=5,
        stride=1,
        hidden_units=12,
        labels_dim=5,
        L2_beta=0.001,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer
    )
    model.train(
        loss_function="MSE",
        optimizer=adam
    )
