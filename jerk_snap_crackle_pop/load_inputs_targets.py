import os
from collections import deque
import numpy as np
import pandas as pd

# "T_max" and "features" are both hard-coded for now
# T_max = 1849=43*43 allows us to map the inputs onto a square grid
# so as to formulate the problem via an image recognition analogy
T_max = 43**2
features = ['x', 'y', 'v_x', 'v_y', 'accl_x', 'accl_y','jerk_x',\
            'jerk_y','snap_x','snap_y','crak_x','crak_y','pop_x','pop_y']
num_channels = len(features)

def _load_trip_features(trip_data,features=features,T_max=T_max):

    """ Load preprocessed data ("trip_data") from a single trip."""
    df = pd.read_pickle(trip_data)
    trip_features = df[features].values.transpose()
    # make sure there are no NaNs
    assert(not np.any(np.isnan(trip_features)))
    assert(not np.any(np.isinf(trip_features)))
    # "standardize" data by subtracting mean & re-scaling by std. deviation
    trip_features = trip_features - np.mean(trip_features, axis=0)
    trip_features = trip_features / np.std(trip_features, axis=0)
    # replace any NaNs (that may be caused by previous division) with zeros
    trip_features = np.nan_to_num(trip_features)
    # get final x,y coordinates of trip
    x_T,y_T = trip_features[(0,1),-1]
    # fix the shape of the data so that all inputs have length T_max
    fit_size= np.zeros((trip_features.shape[0],
                        T_max-trip_features.shape[1])).astype(trip_features.dtype)
    # freeze the x,y coordinates for all t>data.shape[1]
    fit_size[0]=x_T*np.ones((1,fit_size.shape[1])).astype(fit_size.dtype)
    fit_size[1]=y_T*np.ones((1,fit_size.shape[1])).astype(fit_size.dtype)
    trip_features = np.concatenate((trip_features, fit_size),axis=1)
    return trip_features.reshape((1, trip_features.shape[0],
                                  trip_features.shape[1]))


def add_features_to_driver(drvr_data_loc, runtime_configuration, split_data=True):
    """ Collate trip_features associated with a given driver."""
    train_val_ratio = runtime_configuration['train_val_ratio']
    assert(train_val_ratio[0]+train_val_ratio[1]==1)
    driver_feats = deque([])
    for fname in os.listdir(drvr_data_loc):
        if os.path.isfile(os.path.join(drvr_data_loc, fname)):
            driver_feats.append(
            _load_trip_features(os.path.join(drvr_data_loc, fname)))
    # driver_feats contains K objects of shape (1,num_channels,T_max)
    # D_i will be numpy array of shape (K,num_channels,T_max)
    # associated with all K trips made by driver i
    D_i = np.concatenate(driver_feats)
    drvr_IDs = np.ones(D_i.shape[0]) * int(os.path.basename(drvr_data_loc))
    if split_data:
        splitoff = int(np.floor(train_val_ratio[0] * drvr_IDs.shape[0]))
        return dict(train_with=dict(D_i=D_i[:splitoff],
                    drvr_IDs=drvr_IDs[:splitoff]),
                    val_with=dict(D_i=D_i[splitoff:],
                    drvr_IDs=drvr_IDs[splitoff:]))
    else:
        return dict(D_i=D_i, drvr_IDs=drvr_IDs)


def _finalize_data(all_training_targets, all_val_targets):
    unique_val_targets = np.unique(all_val_targets)
    cloned_training_targets, cloned_val_targets = \
                    np.copy(all_training_targets), np.copy(all_val_targets)

    for lbl in range(unique_val_targets.shape[0]):
        cloned_training_targets[all_training_targets == unique_val_targets[lbl]] = lbl
        cloned_val_targets[all_val_targets == unique_val_targets[lbl]] = lbl
    return cloned_training_targets, cloned_val_targets, unique_val_targets



def set_inputs_n_targets(data_logs, runtime_configuration):
    """ Prepare complete network-ready inputs."""
    trips_per_drvr = 200 # ugh! TODO:remove hard-coding for trips_per_drvr
    drivers_dir = data_logs['data_location']
    training_ratio=runtime_configuration['train_val_ratio'][0]
    drvr_list = [drvr for drvr in os.listdir(drivers_dir)
                if os.path.isdir(os.path.join(drivers_dir, drvr))]
    packaged_data = dict(train_with=dict(allDs=None,drvr_IDs=None),
                     val_with=dict(allDs=None,drvr_IDs=None))

    # TODO: add num_channels, T_max as parameters passed to the function
    num_training_trips = int(np.floor(training_ratio * trips_per_drvr))
    num_val_trips = int(trips_per_drvr - num_training_trips)
    total_train_samples = len(drvr_list) * num_training_trips
    total_val_samples = len(drvr_list) * num_val_trips
    assert(total_train_samples + total_val_samples == trips_per_drvr * len(drvr_list))

    # create helper functions for slicing to aid in splitting the inputs into
    # training and validation sets
    bunch_train_trips = \
        lambda k: slice(k * num_training_trips, (k + 1) * num_training_trips)
    bunch_val_trips = \
        lambda k: slice(k * num_val_trips, (k + 1) * num_val_trips)

    # initialize data structures for inputs and targets
    all_training_inputs  = np.zeros(shape=(total_train_samples,num_channels,T_max),
                                   dtype=np.float16)
    all_training_targets = np.zeros(shape=(total_train_samples), dtype=np.int16)

    all_val_inputs  = np.zeros(shape=(total_val_samples,num_channels,T_max),
                              dtype=np.float16)
    all_val_targets = np.zeros(shape=(total_val_samples), dtype=np.int16)


    # populate data structures for inputs and targets
    for i, drvr in enumerate(drvr_list):
        driver_feats = add_features_to_driver(os.path.join(drivers_dir, drvr),\
                                               runtime_configuration)
        all_training_inputs[bunch_train_trips(i)] =\
           driver_feats['train_with']['D_i'].astype(all_training_inputs.dtype)
        all_training_targets[bunch_train_trips(i)]=\
           driver_feats['train_with']['drvr_IDs'].astype(all_training_targets.dtype)
        all_val_inputs[bunch_val_trips(i)] =\
            driver_feats['val_with']['D_i'].astype(all_val_inputs.dtype)
        all_val_targets[bunch_val_trips(i)] =\
            driver_feats['val_with']['drvr_IDs'].astype(all_val_targets.dtype)
        print('{0} % of drivers processed ...'.format(np.round(100.0*(i+1)/len(drvr_list))))

    # package inputs and targets into a dictionary
    packaged_data['train_with']['allDs']=all_training_inputs
    packaged_data['val_with']['allDs']=all_val_inputs

    packaged_data['train_with']['drvr_IDs'], \
              packaged_data['val_with']['drvr_IDs'], \
                      packaged_data['class_labels'] \
                        = _finalize_data(all_training_targets, all_val_targets)

    return packaged_data
