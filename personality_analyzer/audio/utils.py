import matplotlib.pyplot as plt
import numpy as np
import os


def save_first_layer_filters_plot(model, savedir='figures/first_layer_filter_parameters',
                                  sampling_freq=8000):
    """
    Saves subplots of filter parameters without grid and axes.
    :return:
    """
    conv_layer_weights = model.layers[1].get_weights()[0]
    conv_layer_biases = model.layers[1].get_weights()[1]
    print('Weights', conv_layer_weights.shape)
    print('Biases ', conv_layer_biases.shape)

    conv_layer_R = conv_layer_weights[:, 0, :].T
    print('Weights', conv_layer_R.shape)

    # conv_layer_E = conv_layer_weights[:, 1, :].T

    patch_size = conv_layer_R.shape[1]  # e.g. patch size of 200 at 8 kHz sampling rate is 25 ms in size

    xtime = np.linspace(0, patch_size / sampling_freq * 1000, patch_size)
    plt.xlabel('time (ms)')

    # plt.savefig('figures/'+save_date+'-first layer filter parameters.eps')
    # plt.show()

    # Subplots.
    filt1 = plt.subplot(341)
    plt.plot(xtime, conv_layer_R[0, :])
    filt1.axis('off')

    filt2 = plt.subplot(342)
    plt.plot(xtime, conv_layer_R[1, :])
    filt2.axis('off')

    filt3 = plt.subplot(343)
    plt.plot(xtime, conv_layer_R[2, :])
    filt3.axis('off')

    filt4 = plt.subplot(344)
    plt.plot(xtime, conv_layer_R[3, :])
    filt4.axis('off')

    filt5 = plt.subplot(345)
    plt.plot(xtime, conv_layer_R[4, :])
    filt5.axis('off')

    filt6 = plt.subplot(346)
    plt.plot(xtime, conv_layer_R[5, :])
    filt6.axis('off')

    filt7 = plt.subplot(347)
    plt.plot(xtime, conv_layer_R[6, :])
    filt7.axis('off')

    filt8 = plt.subplot(348)
    plt.plot(xtime, conv_layer_R[7, :])
    filt8.axis('off')

    filt9 = plt.subplot(349)
    plt.plot(xtime, conv_layer_R[8, :])
    filt9.axis('off')

    filt10 = plt.subplot(3, 4, 10)
    plt.plot(xtime, conv_layer_R[9, :])
    filt10.axis('off')

    filt11 = plt.subplot(3, 4, 11)
    plt.plot(xtime, conv_layer_R[10, :])
    filt11.axis('off')

    filt12 = plt.subplot(3, 4, 12)
    plt.plot(xtime, conv_layer_R[11, :])
    filt12.axis('off')

    plt.savefig(os.path.join(savedir, 'first_layer_filter_parameters.eps'))
