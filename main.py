"""
    Dependencies: Python3, Pandas, Theano, Lasagne
    Pipeline/Flow:
    Assumes that preprocessed data exists and is 
    located in data_logs['data_location'] as 
    specified in configurations.py
"""

import time
import os
import pickle
import numpy as np
from lasagne.layers import get_all_param_values,noise

from configurations import data_logs, runtime_configuration
from load_inputs_targets import set_inputs_n_targets
from helper_funcs import train_using
from networks import conv_net_X


def save_it(model, data_logs,runtime_configuration):
    """ Save trained model in data_logs['record_stats_location']."""
    np.random.seed(runtime_configuration['randseed'])
    stamp_it = np.random.randint(100)
    locate_it  = data_logs['record_stats_location']
    savepath = os.path.join(locate_it, 'model_{0}.pkl'.format(stamp_it))
    os.makedirs(os.path.dirname(savepath), exist_ok=True)
    with open(savepath, 'wb') as f:
        pickle.dump(model, f)
    return




def train_model(data_logs, runtime_configuration):

    print("********************************************************")
    print("Setting up inputs & targets. This could take a while ...")
    print("")
    packaged_data = set_inputs_n_targets(data_logs, runtime_configuration)

    neural_net = conv_net_X(runtime_configuration,\
                 num_output_units=len(packaged_data['class_labels']))
    print("********************************************************")
    print("Now training ...")
    tick = time.time()
    try:
        train_using(packaged_data['train_with'], packaged_data['val_with'],\
                     neural_net, data_logs, runtime_configuration)
        print('Finished training without interruption ...')
    except KeyboardInterrupt:
        print('Terminating ...')
    tock = time.time()

    saved_to = os.path.abspath(data_logs['record_stats_location'])
    save_it(model=get_all_param_values(neural_net), data_logs=data_logs,\
                                    runtime_configuration=runtime_configuration)


    mins, secs = divmod(int(tock-tick), 60)
    hrs, mins = divmod(mins, 60)

    print('===============================')
    print('   Network\'s "state" saved to '
              ' "{0}".\n'
          '===============================\n'
          'Training time: {1[0]:02d} hrs, {1[1]:02d} mins, {1[2]:02d} secs  \n'
     .format(saved_to, (hrs,mins,secs)))


    return neural_net

def main():
    np.random.seed(runtime_configuration['randseed'])
    noise._srng = noise.RandomStreams(runtime_configuration['randseed'])
    train_model(data_logs, runtime_configuration)
    return


if __name__ == '__main__':
    main()
