import theano

from theano_ctc.cpu_ctc import cpu_ctc_cost
from theano_ctc.gpu_ctc import gpu_ctc_cost
from theano.sandbox.cuda.type import CudaNdarrayType
from theano.sandbox.cuda.basic_ops import gpu_from_host
import os
from ctypes import cdll

cdll.LoadLibrary(os.path.join(os.environ["CTC_LIB"], "build", "libwarpctc.so"))

def ctc_cost(acts, labels, input_lengths = None):
  # This should be properly integrated into the theano optimization catalog.
  # Until then, this forces the choice based on device configuration.
  if theano.config.device.startswith("gpu") or theano.sandbox.cuda.cuda_enabled:
    if not isinstance(acts.type, CudaNdarrayType): # if not already on the device
      acts = gpu_from_host(acts)  # this should get optimized away
    return gpu_ctc_cost(acts, labels, input_lengths)
  else:
    return cpu_ctc_cost(acts, labels, input_lengths)

