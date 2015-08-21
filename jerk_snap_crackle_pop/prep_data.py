""" Objective: Extract features from data
    Assumptions:
    (0) Entire dataset is contained in a directory named "data/drivers"
    (1) Dataset represents trips from N=2730 drivers stored in "data/drivers"
    (2) Trips made by driver "i" are stored in the subdirectory "data/drivers/i"
    (4) Each subdirectory "data/drivers/i" contains K=200 csv files corresponding
        to driver i's trip data. Let the files be labeled by "k=1:K"
    (5) Each trip "k" consists of a pair of time-dependent coordinates
        (x(t),y(t)) in time-steps of length dt=1 second
    (6) We first go through the entire dataset and extract the duration of
        the longest trip, which we call Tmax
    (7) Suppose that the "kth" trip made by driver "i" lasts for T_(i,k) seconds
        with  T_(i,k) < Tmax ...
        ... then we set x(t),y(t)=x(T_(i,k)),y(T_(i,k)) for all t>T_(i,k)
        This ensures that all trips have the same length.

    Features:
    (0) For each trip, we will use the kinematic feature tuple
        (position, velocity, acceleration, jerk, snap, crackle, pop).
    (1) Each feature in the tuple has an x- and a y-component.
    (2) Each feature is a time derivative of the previous feature in the tuple.
    (3) This gives a total of 14 features.

    Strategy:
    (0) Use pandas' dataframes to load the raw data
    (1) Define functions which extract the features
    (2) Define
        (i)   a function which adds features to a single trip
        (ii)  a function which adds features to all trips by a single driver
        (iii) a function which adds features to all trips by all drivers

        The third function repeatedly calls the second function, while
        the second function repeatedly calls the first function.
    (3) Use multiprocessing to run multiple processes

    """
import os
import errno
import warnings
import multiprocessing

import numpy as np
import pandas as pd


num_procs = 4
locus = {"raw_data":'raw_data/data/drivers',\
           "preprocessed": 'preprocessed/data/drivers'}


try:
    os.makedirs(locus['preprocessed'])
except OSError as exception:
    if exception.errno == errno.EEXIST:
        warnings.warn('{0} already exists. \
                       Files in it may be \
                       overwritten'.format(locus['preprocessed']))
    else:
        raise



def _load_trip(csv_file, dtype=np.float64):
    """ Assumes data from a single trip stored in csv_file."""
    df = pd.DataFrame.from_csv(path=csv_file, index_col=None).astype(dtype)
    np.testing.assert_equal(df.columns, np.array(['x', 'y']))
    # remove any NaNs that may exist
    df.interpolate()
    assert(not np.any(np.isnan(df.values)))
    return df


def _calc_trip_velocities(df):
    x, y = df['x'], df['y']
    dt = 1
    df['v_x'] = np.gradient(x, dt)
    df['v_y'] = np.gradient(y, dt)
    return


def _calc_trip_accels(df):
    v_x, v_y = df['v_x'], df['v_y']
    dt = 1
    df['accl_x'] = np.gradient(v_x, dt)
    df['accl_y'] = np.gradient(v_y, dt)
    return

def _calc_trip_jerks(df):
    a_x, a_y = df['accl_x'], df['accl_y']
    dt = 1
    df['jerk_x'] = np.gradient(a_x, dt)
    df['jerk_y'] = np.gradient(a_y, dt)
    return

def _calc_trip_snaps(df):
    j_x, j_y = df['jerk_x'], df['jerk_y']
    dt = 1
    df['snap_x'] = np.gradient(j_x, dt)
    df['snap_y'] = np.gradient(j_y, dt)
    return

def _calc_trip_crackles(df):
    s_x, s_y = df['snap_x'], df['snap_y']
    dt = 1
    df['crak_x'] = np.gradient(s_x, dt)
    df['crak_y'] = np.gradient(s_y, dt)
    return

def _calc_trip_pops(df):
    c_x, c_y = df['crak_x'], df['crak_y']
    dt = 1
    df['pop_x'] = np.gradient(c_x, dt)
    df['pop_y'] = np.gradient(c_y, dt)
    return


def _calc_features(df):
    _calc_trip_velocities(df)
    _calc_trip_accels(df)
    _calc_trip_jerks(df)
    _calc_trip_snaps(df)
    _calc_trip_crackles(df)
    _calc_trip_pops(df)
    return


def add_features_to_trip(source_file, target_file):
    """ Adds features to a single trip."""
    if source_file == target_file:
        raise ValueError('Overwriting source files is not allowed.')
    df = _load_trip(source_file)
    _calc_features(df)
    if not os.path.exists(os.path.dirname(target_file)):
        os.makedirs(os.path.dirname(target_file))
    df.astype(np.float32).to_pickle(target_file)
    return


def add_features_to_all_trips_by_single_driver(source_dir, target_dir):
    """ Add features to all trips associated with a single driver."""
    if source_dir == target_dir:
        raise ValueError('Overwriting source directories is not allowed.')
    files = [f for f in os.listdir(source_dir)
             if os.path.isfile(os.path.join(source_dir, f))]
    source = lambda f: os.path.join(source_dir, f)
    target = lambda f: os.path.join(target_dir, os.path.splitext(f)[0] + '.pkl')
    for f in files:
        add_features_to_trip(source(f), target(f))
    return


def add_features_to_entire_dataset(source_dir, target_dir, multiproc_poolsize=1):
    """ Add features to all trips in the dataset."""
    if source_dir == target_dir:
        raise ValueError('Overwriting source directories is not allowed.')
    subdirs = [s for s in os.listdir(source_dir)
                   if os.path.isdir(os.path.join(source_dir, s))]
    pool = multiprocessing.Pool(multiproc_poolsize)
    source_subdir = lambda subd: os.path.join(source_dir, subd)
    target_subdir = lambda subd: os.path.join(target_dir, subd)
    pool.starmap(add_features_to_all_trips_by_single_driver, \
                 [(source_subdir(subd), target_subdir(subd)) for subd in subdirs])
    return


def main():
    add_features_to_entire_dataset(source_dir=locus['raw_data'],\
                                   target_dir=locus['preprocessed'], \
                                   multiproc_poolsize=num_procs)

if __name__ == '__main__':
    main()
