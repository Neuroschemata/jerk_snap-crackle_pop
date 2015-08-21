import os
import numpy as np
import theano
import theano.tensor as T
import lasagne

def record_performance(live_performance,data_logs,runtime_configuration):
    field_names = 'epoch \t loss \t correct predictions (train) \t correct predictions (validate) \n' \
                  '----- \t ----- \t -------------------------- \t ------------------------------ \n'
    field = '{0} \t {1} \t {2} \t {3} \n'.format(\
    live_performance['epoch'], \
    live_performance['loss'],\
    live_performance['precision_t'],\
    live_performance['recall_t'])
    os.makedirs(data_logs['record_stats_location'], exist_ok=True)
    np.random.seed(runtime_configuration['randseed'])
    stamp_it = np.random.randint(100)
    filename = os.path.join(data_logs['record_stats_location'], \
    'performance_{0}.log'.format(stamp_it))
    with open(filename, 'a') as performance:
        if performance.tell() == 0:
            performance.write(field_names)
        performance.write(field)
    return

def display_live_stats(live_performance):
    """ Print live performance to stdout """
    print('Epoch #{0}\n'
          '  observed loss:\t\t\t\t{1:.3f}\n'
          '  observed accuracy on training data:\t\t{2:.4f} \n'
          '  observed accuracy on validation data:\t\t{3:.4f} '
          .format(live_performance['epoch'], live_performance['loss'], \
                  live_performance['precision_t'],\
                  live_performance['recall_t']))
    return



def _metric_func_pair(network, eta=0.01, momentum=0.9, minibatch_size=100):
    """ set up a pair of Theano functions to evaluate metrics."""
    X = T.ftensor3('X')
    y = T.ivector('y')
    # determinisic = False/True turns dropout ON/OFF
    train_phase_outputs = lasagne.layers.get_output(network, X, deterministic=False)
    test_phase_outputs = lasagne.layers.get_output(network, X, deterministic=True)
    loss = T.sum(T.nnet.categorical_crossentropy(train_phase_outputs, y))
    prediction = T.argmax(test_phase_outputs, axis=1)
    accuracy = T.sum(T.eq(prediction, y), dtype=theano.config.floatX)
    params = lasagne.layers.get_all_params(network, trainable=True)
    updates = lasagne.updates.nesterov_momentum(loss, params, eta, momentum)
    eval_loss = theano.function(inputs=[X, y], outputs=loss, updates=updates)
    eval_accuracy = theano.function(inputs=[X, y], outputs=accuracy)
    return eval_loss, eval_accuracy


def _input_slabs(inputs_dict, runtime_configuration, shuffle=True):
    """ Make input slabs to aid parallelization on GPU."""
    slab_size = runtime_configuration['gpu_slab_size']
    assert(inputs_dict['allDs'].shape[0] == inputs_dict['drvr_IDs'].shape[0])
    input_size = inputs_dict['drvr_IDs'].shape[0]
    indx_range = np.arange(input_size).astype(np.int)
    assert(input_size == indx_range.shape[0])
    if shuffle:
        indx_range = np.random.permutation(indx_range)
    for i in range(0, input_size, slab_size):
        sub_interval = slice(i, i + slab_size)
        yield dict(
            slab_Ds=inputs_dict['allDs'][indx_range[sub_interval]],
            slab_drvrs=inputs_dict['drvr_IDs'][indx_range[sub_interval]],
            num_drvrs_in_slab=indx_range[sub_interval].shape[0]
        )


def _run_thru_slabs_n_batches(inputs_dict, eval_loss, eval_accuracy, \
                                runtime_configuration, train_flag=False):
    """ Process slabs and minibatches """
    minibatch_size =runtime_configuration['minibatch_size']
    loss = accuracy = count= 0
    
    for slab in _input_slabs(inputs_dict, runtime_configuration, shuffle=train_flag):
        for i in range(0, slab['num_drvrs_in_slab'], minibatch_size):
            batch_interval = slice(i, i + minibatch_size)
            if train_flag:
                loss += eval_loss(slab['slab_Ds'][batch_interval].astype(np.float32),
                                 slab['slab_drvrs'][batch_interval].astype(np.int32))
            accuracy += eval_accuracy(slab['slab_Ds'][batch_interval].astype(np.float32),
                           slab['slab_drvrs'][batch_interval].astype(np.int32))
            count += slab['slab_drvrs'][batch_interval].shape[0]

    assert(count == inputs_dict['drvr_IDs'].shape[0])
    loss = loss / count
    accuracy = accuracy / count
    return loss, accuracy


def train_using(inputs_dict_trn, inputs_dict_val, network,\
                                 data_logs,runtime_configuration):
    """ Train for num_epochs."""
    eval_loss, eval_accuracy = \
        _metric_func_pair(network, runtime_configuration['eta'], \
                                  runtime_configuration['momentum'], \
                                  runtime_configuration['minibatch_size'])

    for epoch in range(runtime_configuration['num_epochs']):
        loss, precision_t = _run_thru_slabs_n_batches(inputs_dict_trn,\
                                   eval_loss, eval_accuracy, \
                                   runtime_configuration, train_flag=True)
        _, recall_t = _run_thru_slabs_n_batches(inputs_dict_val,\
                                  eval_loss, eval_accuracy, \
                                  runtime_configuration, train_flag=False)
        live_performance = dict(epoch=epoch, loss=loss, \
                                precision_t=precision_t,\
                                recall_t=recall_t)
        if data_logs['log_stats']:
            record_performance(live_performance, data_logs,runtime_configuration)
        if data_logs['live_stats']:
            display_live_stats(live_performance)
    return network
