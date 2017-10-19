from keras import backend


def get_input_shape(image_size, num_channels):
    return ((num_channels,) + tuple(image_size)) if backend.image_data_format() == 'channels_first' \
        else (tuple(image_size) + (num_channels,))


def get_channel_axis():
    return 1 if backend.image_data_format() == 'channels_first' else -1
