import matplotlib.pyplot as plt
import numpy as np
import os
from scipy import signal


def save_first_layer_filters_plot(model, savedir='figures/first_layer_filter_parameters',
                                  sampling_freq=8000):
    """
    Saves subplots of filter parameters without grid and axes.
    """
    conv_layer_weights, conv_layer_biases = _get_weights_and_biases(model)

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


def first_layer_fft(model, sampling_freq : int = 8000):

    conv_layer_weights, conv_layer_biases = _get_weights_and_biases(model)

    conv_layer_R = conv_layer_weights[:, 0, :].T  # (n_filters, kernel_size)
    conv_layer_E = conv_layer_weights[:, 1, :].T

    num_filters = conv_layer_R.shape[0]

    four_log_R = _get_frequency_components(conv_layer_R)
    four_log_E = _get_frequency_components(conv_layer_E)

    # Average filter response first convolutional layer.
    xfreq = np.linspace(0, sampling_freq / 2, 101)
    plt.subplot(211)
    yfreq = np.mean(four_log_R, axis=0)
    # plt.plot(xfreq, yfreq)
    plt.plot(xfreq, signal.savgol_filter(yfreq, 9, 5))
    # plt.title('Average over filters |FFT|')
    plt.grid()
    plt.ylabel('20 log10(|FFT|))')

    plt.subplot(212)
    yfreq = np.mean(four_log_E, axis=0)
    # plt.plot(xfreq, yfreq)
    plt.plot(xfreq, signal.savgol_filter(yfreq, 9, 5))
    plt.xlabel('Frequency')
    plt.grid()
    plt.ylabel('20 log10(|FFT|))')

    plt.savefig('average-filters-personality.eps')
    plt.show()

    # Response per filter in frequency domain (i.e. sorted image).
    filter_im_R = four_log_R.T
    print(filter_im_R.shape)
    filter_max_R = np.max(filter_im_R)
    filter_min_R = np.min(filter_im_R)
    print('Max value ', np.max(filter_im_R), 'Min value', np.min(filter_im_R))

    filter_maxima_R = np.argsort(np.argmax(filter_im_R, axis=0))
    filter_im_R = filter_im_R[:, filter_maxima_R]

    filter_im_E = four_log_E.T
    print(filter_im_E.shape)
    filter_max_E = np.max(filter_im_E)
    filter_min_E = np.min(filter_im_E)
    print('Max value ', np.max(filter_im_E), 'Min value', np.min(filter_im_E))

    filter_maxima_E = np.argsort(np.argmax(filter_im_E, axis=0))
    filter_im_E = filter_im_E[:, filter_maxima_E]

    plt.imshow(
        filter_im_R,
        aspect='auto',
        cmap='seismic',
        vmin=-filter_max_R,
        vmax=filter_max_R,
        extent=(0, num_filters, 4000, 0)
    )
    # plt.colorbar()
    plt.ylabel('frequency')
    plt.tight_layout()
    plt.savefig('pers-filter-responses-raw.eps')
    plt.title('Raw waveforms')
    plt.show()
    plt.imshow(
        filter_im_E,
        aspect='auto',
        cmap='seismic',
        vmin=-filter_max_E,
        vmax=filter_max_E,
        extent=(0, num_filters, 4000, 0)
    )
    plt.ylabel('frequency')
    plt.tight_layout()
    plt.savefig('pers-filter-responses-energy.eps')
    plt.show()

    # plt.subplot(211)
    # plt.imshow(filter_im_R, aspect='auto', cmap='seismic', vmin=-filter_max_R, vmax=filter_max_R)
    # plt.colorbar()
    # plt.subplot(212)
    # plt.imshow(filter_im_E, aspect='auto', cmap='seismic', vmin=-filter_max_E, vmax=filter_max_E)
    # plt.colorbar()
    # plt.savefig('frequency-response-per-filter-pers.eps')
    # plt.show()


def _get_weights_and_biases(model):
    conv_layer_weights = model.layers[1].get_weights()[0]  # (kernel_size, n_input_channels, n_filters)
    conv_layer_biases = model.layers[1].get_weights()[1]  # (n_filters, )
    print('Weights', conv_layer_weights.shape)
    print('Biases ', conv_layer_biases.shape)
    return conv_layer_weights, conv_layer_biases


def _get_frequency_components(conv_layer):
    four = np.fft.rfft(conv_layer)  # (n_filters, kernel_size / 2 + 1)
    print(four.shape)
    four_log = 20 * np.log10(np.abs(four))
    print(four_log.shape)
    return four_log
