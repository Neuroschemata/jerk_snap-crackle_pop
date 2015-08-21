"""
<<<<<<< HEAD
N=2730,K=200,gpu_slab_size=N*K//50
runtime_configuration stores the network parameters
data_logs stores info pertaining to data loading and saving, 
as well as live displays.
"""
runtime_configuration = {'train_from_scratch': True, \
                          'path_to_pretrained_model':'',\
=======
- N=2730,K=200,gpu_slab_size=N*K//50
- runtime_configuration stores the network parameters
- data_logs stores info pertaining to data loading and saving,
  as well as live displays.
"""
runtime_configuration = { 'num_epochs':10,\ # set to 10 for demo purposes
                          'eta':0.01,\
                          'momentum':0.9,\
                          'dropout_fraction':0.2,\
                          'randseed':1,\
>>>>>>> b207544ce120895082427d0bd92a013450066c33
                          'train_val_ratio':(0.8,0.2),\
                          'minibatch_size':20,\
                          'batch_normalization':True,\
                          'gpu_slab_size':10920,\
                          'num_conv_filters':128,\
                          'num_units_dense_layer':128,\
<<<<<<< HEAD
                          'num_epochs':2,\
                          'eta':0.01,\
                          'momentum':0.9,\
                          'dropout_fraction':0.2,\
                          'randseed':1}
=======
                          'train_from_scratch': True, \
                          'path_to_pretrained_model':''}

>>>>>>> b207544ce120895082427d0bd92a013450066c33
data_logs = {'data_location':'preprocessed/data/drivers',\
             'record_stats_location': 'results/runs',\
             'log_stats':True, 'live_stats':True}
