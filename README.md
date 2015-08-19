jerk, snap, crackle & pop
=========================

The terms jerk, snap, crackle and pop respectively refer to the
first, second, third and fourth derivatives of acceleration, and
are a crucial component of the features extracted to solve the telematics
problem posed [here](https://www.kaggle.com/c/axa-driver-telematics-analysis).

Details about the solution and the code is available in `doc/documentation.pdf`.
The remainder of this document describes how to use the code.

## Quick Guide To The Code

**Make sure you have all the dependencies installed**

Requires Python3, Pandas, Numpy, Theano, and Lasagne.

Instructions for installing Theano and getting it to run on a GPU can be found [here](http://deeplearning.net/software/theano/install.html).

Instructions for installing Lasagne can be found [here](http://lasagne.readthedocs.org/en/latest/user/installation.html).

### Getting The Data
Make sure you have sufficient space to store the data. The zipped data
requires 1.5GB, and unpacking it requires an additional 6GB.

Download the zipped data from [Kaggle](https://www.kaggle.com/c/axa-driver-telematics-analysis/data). Place and extract the files in the folder `raw_data`

### Preprocess The Data

**This step extracts features from the data and pickles it.**

* Assuming the data is stored in `raw_data`, run `python3 prep_data.py`
to extract features from the data.
* The results will be stored in a
folder named 'preprocessed/data/drivers'.
* You may choose to specify where to store the "raw data" and the
"preprocessed data" in the dictionary `locus` at the top of the file `prep_data.py`.

### Load data into RAM and train the network
* Once the pre-processing step is complete, everything else is done in one step
by running `python3 main.py`.
* The training parameters can be set in the configuration file `configurations.py`.


|  Parameter        | Description |
|:------------------|:------------|
| `train_from_scratch`     | If `True`, trains the network using randomly initialized parameters. If `False`, trains using parameters from a previous training run, assumed to be located in `path_to_pretrained_model`.|
|`path_to_pretrained_model`|specifies the location of pre-trained network parameters.|
|`train_val_ratio`| specifies the ratio of the size of the training data to the size of the validation data.|
|`minibatch_size`| specifies the size of the mini-batches used during training.|
|`batch_normalization` | If `True`, turns on batch normalization.|
|`gpu_slab_size`|specifies the size of the loads to be parallelized.|
|`num_conv_filters`|specifies the number of convolution filters.|
|`num_units_dense_layer`| specifies the number of units in the dense layers.|
|`num_epochs`|species the number of complete passes through the data.|
|`eta `| specifies the learning rate.|
|`momentum`| specifies the value for the momentum.|
|`dropout_fraction`|species the fraction of units randomly picked by "dropout".|
|`randseed`| species the seed for the random-number-generator used.|
### Observations/Afterthoughts
1. Lack of support for multiple GPUs.
 * As written, the code takes advantage of a single
 GPU which is relatively slow even on a problem of this size.
 * I need to learn how to utilize multiple GPUs to speed up training. 
* The main approach here was to map the original problem onto an image recognition/classification problem
so as to take advantage of convnets. I now wonder whether there is some advantage to be had in formulating
the problem as a sequence prediction task, utilizing bi-directional RNNs with LSTMs/CTC instead.
