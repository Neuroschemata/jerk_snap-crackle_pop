#!/usr/bin/env python                                                                                   
# ----------------------------------------------------------------------------                          
# Copyright 2015 Nervana Systems Inc. All rights reserved.                                              
# Unauthorized copying or viewing of this file outside Nervana Systems Inc.,                            
# via any medium is strictly prohibited. Proprietary and confidential.                                  
# ----------------------------------------------------------------------------                          
import platform
import os
import numpy as np
from cffi import FFI
from neon.backends import gen_backend

ffi = FFI()

header = """
typedef struct {
int loc; 
unsigned int num_threads;
void* stream;
int blank_label;
}ctcOptions;

int get_workspace_size(const int* const label_lengths,
                       const int* const input_lengths,
                       int alphabet_size, 
                       int minibatch,
                       ctcOptions options,
                       size_t* size_bytes);

int compute_ctc_loss(const float* const activations,
                     float* gradients,
                     const int* const flat_labels,
                     const int* const label_lengths,
                     const int* const input_lengths,
                     int alphabet_size,
                     int minibatch,
                     float *costs,
                     void *workspace, 
                     ctcOptions options);
"""

def get_data():
    alphabet = '_ABCD'
    max_t, max_s, nout = 2, 2, len(alphabet)
    bsz = 1
    activations = np.zeros((nout, max_t))
    activations[:, 0] = [0.1, 0.6, 0.1, 0.1, 0.1]
    activations[:, 1] = [0.1, 0.1, 0.6, 0.1, 0.1]
    activations = activations.reshape((nout, max_t, bsz))
    activations = np.array(activations)
    _activations = np.zeros((max_t, bsz, nout))
    _activations = np.transpose(activations, (1,2,0))
    _grads = np.zeros(_activations.shape)
    costs = np.zeros(bsz)
    labels = np.zeros((1, 2), dtype=np.int32)
    labels[0, 0] = 1
    labels[0, 1] = 2
    input_lens = np.array(np.array([max_t]), dtype=np.int32)
    label_lens = np.array(np.array([max_s]), dtype=np.int32)
    return _activations, labels, input_lens, label_lens

def get_buf_size(ptr_to_buf):
    return ffi.sizeof(ffi.getctype(ffi.typeof(ptr_to_buf).item))

def ctc_cpu(acts, lbls, utt_lens, lbl_lens, n_threads=8):
    CTC_CPU = 0 
    blank_label = 0
    libpath = os.path.join("warp-ctc", "build", "libwarpctc.so")
    ctclib = ffi.dlopen(libpath)
    assert os.path.isfile(libpath), ("Expected libwarpctc.so at {} but not found. "
                                     "Try running make").format(libpath)
    ffi.cdef(header)
    size_in_bytes = ffi.new("size_t*")
    options = ffi.new('ctcOptions*', 
                      {"loc": CTC_CPU, 
                      "num_threads": n_threads, 
                      "blank_label": blank_label})[0]
    max_t, bsz, nout = acts.shape
    flat_act_dims = np.prod(acts.shape)
    flat_grad_dims = flat_act_dims
    costs = np.zeros((bsz), dtype=np.float32)
    grads = np.zeros((flat_grad_dims), dtype=np.float32)
    warp_costs = ffi.new("float[]", bsz)
    warp_grads = ffi.new("float[]", flat_grad_dims)
    warp_grads_buf_size = flat_grad_dims * get_buf_size(warp_grads)
    warp_costs_buf_size = bsz * get_buf_size(warp_costs)

    warp_acts = ffi.new("float[]", flat_act_dims)
    warp_acts[0:flat_act_dims] = acts.ravel()

    warp_labels = ffi.cast("int*", lbls.ravel().ctypes.data)
    warp_label_lens = ffi.cast("int*", lbl_lens.ravel().ctypes.data)
    warp_input_lens = ffi.cast("int*", utt_lens.ravel().ctypes.data)

    status = ctclib.get_workspace_size(warp_label_lens, warp_input_lens, nout, bsz, options, size_in_bytes)
    workspace = ffi.new("char[]", size_in_bytes[0])
    ctc_status = ctclib.compute_ctc_loss(warp_acts,
                                         warp_grads,
                                         warp_labels,
                                         warp_label_lens,
                                         warp_input_lens,
                                         nout, 
                                         bsz, 
                                         warp_costs,
                                         workspace, 
                                         options);

    ffi.memmove(grads, warp_grads, warp_grads_buf_size)
    grads = grads.reshape((acts.shape))
    ffi.memmove(costs, warp_costs, warp_costs_buf_size)

    print("cost = {}".format(costs.ravel()[0]))
    print("grads = {}".format(grads.ravel()))

acts, lbls, utt_lens, lbl_lens = get_data()
ctc_cpu(acts, lbls, utt_lens, lbl_lens)
