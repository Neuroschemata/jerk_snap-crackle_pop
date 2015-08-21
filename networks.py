import pickle
import lasagne
from library.batch_norm import batch_norm

def load_model_from(path):
    """ Load a model from path."""
    with open(path, 'rb') as f:
        return pickle.load(f)


def _load_pretrained_network(network, pretrained_values):
    current_values = lasagne.layers.get_all_param_values(network)
    try:
        # load all parameters except for the output layer
        current_values[:-2] = pretrained_values[:-2]
        lasagne.layers.set_all_param_values(network, current_values)
    except ValueError:
        print('Found mismatch between shapes of current & pretrained values.')
        raise
    return network


def conv_net_X(runtime_configuration, num_output_units=None):
    """ Constructs a network with the following architecture:
         Layer 0:    multi-channel input layer
         Layers 1-4: convolution+pooling layers
         Layers 5-6: dense (fully connected) layers
         Layer 7:    softmax output layer
    """

    if not num_output_units:
        raise ValueError("number of outputs required to be a positive integer")
    num_input_channels = 14
    channel_dimensions = 43**2
    num_conv_pool_layers = 4
    num_dense_layers = 2
    conv_filter_width = 5
    padding = 2
    pool_size = 5
    pool_stride = 5
    dropout_fraction = runtime_configuration['dropout_fraction']
    num_conv_filters = runtime_configuration['num_conv_filters']
    dense_layer_size = runtime_configuration['num_units_dense_layer']
    batch_normalization = runtime_configuration['batch_normalization']
    train_from_scratch = runtime_configuration['train_from_scratch']
    path_to_pretrained_model = runtime_configuration['path_to_pretrained_model']

    network = lasagne.layers.InputLayer(shape=(None, num_input_channels,\
                                                     channel_dimensions))
    network = lasagne.layers.DropoutLayer(network, p=dropout_fraction)

    for i in range(num_conv_pool_layers):
        network = lasagne.layers.Conv1DLayer(network,\
                                             num_filters=num_conv_filters,\
                                             filter_size=conv_filter_width,\
                                             pad=padding,\
                                             nonlinearity=lasagne.nonlinearities.rectify)
        if batch_normalization:
            network = batch_norm(network)
        network = lasagne.layers.MaxPool1DLayer(network, pool_size=pool_size,\
                                                         stride=pool_stride)
        network = lasagne.layers.DropoutLayer(network, p=dropout_fraction)

    for i in range(num_dense_layers):
        network = lasagne.layers.DenseLayer(network, dense_layer_size,\
                                        nonlinearity=lasagne.nonlinearities.rectify)
        if batch_normalization:
            network = batch_norm(network)
        network = lasagne.layers.DropoutLayer(network, p=dropout_fraction)

    network = lasagne.layers.DenseLayer(network, num_output_units,\
                                        nonlinearity=lasagne.nonlinearities.softmax)
    if not train_from_scratch:
        pretrained_values = load_model_from(path_to_pretrained_model)
        network = _load_pretrained_network(network, pretrained_values)
    return network
