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
be = gen_backend(backend='gpu', batch_size=1)

buf_ref_from_array = lambda arr: ffi.from_buffer(ffi.buffer(ffi.cast('void*', arr.ptr), arr.nbytes))
buf_ref_from_ptr =  lambda ptr, size: ffi.from_buffer(ffi.buffer(ptr, size))

# Set up cuda stream                                                                                                                         
if be.stream is None:
    stream_ptr = ffi.cast('void*', 0)
    stream_buf_size = ffi.sizeof(ffi.new_handle(be.stream))
    stream_buf = buf_ref_from_ptr(stream_ptr, stream_buf_size)

libpath = os.path.join("warp-ctc", "build", "libwarpctc.so")
ctclib = ffi.dlopen(libpath)
print('Loaded ctclib from {0}'.format(libpath))


alphabet = '_ABCD'
max_t, max_s, nout = 2, 2, len(alphabet)

bsz = be.bsz

activations = np.zeros((nout, max_t))
activations[:, 0] = [0.1, 0.6, 0.1, 0.1, 0.1]
activations[:, 1] = [0.1, 0.1, 0.6, 0.1, 0.1]
activations = activations.reshape((nout, max_t, bsz))
activations = be.array(activations)
_activations = be.zeros((max_t, bsz, nout))
be.copy_transpose(activations, _activations, (1,2,0))
_grads = be.zeros(_activations.shape)
costs = be.zeros(bsz)

labels = be.zeros((1, 2), dtype=np.int32)
labels[0, 0] = 1
labels[0, 1] = 2
input_lens = be.array(np.array([max_t]), dtype=np.int32)
label_lens = be.array(np.array([max_s]), dtype=np.int32)

ffi.cdef('''
typedef struct {
    int loc;
    unsigned int num_threads;
    void* stream;
    int blank_label;
} ctcOptions;

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
''')
# global constants used by warp-ctc
BLANK_LABEL = 0
CTC_GPU = 1

size_in_bytes = ffi.new("size_t*")
options = ffi.new('ctcOptions*', 
                  {"loc": CTC_GPU, 
                   "stream": stream_buf, 
                   "blank_label": BLANK_LABEL})[0]
warp_activs = ffi.cast("float*", buf_ref_from_array(_activations))
warp_grads = ffi.cast("float*", buf_ref_from_array(_grads))
warp_costs = ffi.cast("float*", buf_ref_from_array(costs))
warp_labels = ffi.new("int[]", labels.get().ravel().tolist())
warp_label_lens = ffi.new("int[]", label_lens.get().ravel().tolist())
warp_input_lens = ffi.new("int[]", input_lens.get().ravel().tolist())

status = ctclib.get_workspace_size(warp_label_lens, 
                                   warp_input_lens, 
                                   nout, 
                                   be.bsz, 
                                   options, 
                                   size_in_bytes)
be.set_scratch_size(int(size_in_bytes[0]))
workspace = be.scratch_buffer(int(size_in_bytes[0]))
workspace_buf = buf_ref_from_ptr(ffi.cast('void*', workspace), 
                                 int(size_in_bytes[0]))

ctc_status = ctclib.compute_ctc_loss(warp_activs,
                              warp_grads,
                              warp_labels,
                              warp_label_lens,
                              warp_input_lens,
                              nout, 
                              be.bsz, 
                              warp_costs,
                              workspace_buf, 
                              options);

assert status is 0, "Warp-CTC run failed"

print costs.get().ravel()
print _grads.get()


