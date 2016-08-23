import theano

from theano_ctc.cpu_ctc import CpuCtc
from theano_ctc.gpu_ctc import GpuCtc
from theano.sandbox.cuda.type import CudaNdarrayType
from theano.sandbox.cuda.basic_ops import gpu_from_host
import os
from ctypes import cdll

cdll.LoadLibrary(os.path.join(os.environ["CTC_LIB"], "build", "libwarpctc.so"))

def ctc_cost(acts, labels, input_lengths = None):
  """
  Given sequences of output layer activations and labels, compute the softmax output at each timestep,
  and then compute the CTC cost of each sequence with respect to its corresponding label sequence.

  :param acts: Tensor of pre-softmax activations, with shape=[maxInputSeqLen, batchSize, targetN],
      where
      maxInputSeqLen >= the length of the longest input sequence.
      batchSize is the number of sequences being simultaneously computed / trained.
      targetN is the number of network outputs (<blank> is always target 0).

  :param labels: Matrix of training labels, with shape=[batchSize, maxOutputSeqLen]. 
      Since <blank> is always output 0, labels should be > 0 (targets) or negative (ignored). 
      maxOutputSeqLen >= the length of the longest target sequence (excluding <blank>s, 
      which CTC alignment adds). Label values < 0 at any location are ignored, 
      so [1], [-1, 1, -1], and [-1, -1, 1] are treated the same.

  :param input_lengths: Vector of input sequence lengths, with shape=[batchSize].
      For sequence s (0 <= s < batchSize), CTC is calculated on acts[0:input_lengths[s], s, :].
      If input_lengths is None, then all sequences in the batch are assumed to have length maxInputSeqLen.

  :return: Vector of CTC costs, with shape=[batchSize]
  """
  # This should be properly integrated into the theano optimization catalog.
  # Until then, this forces the choice based on device configuration.
  if theano.config.device.startswith("gpu") or theano.sandbox.cuda.cuda_enabled:
    if not isinstance(acts.type, CudaNdarrayType): # if not already on the device
      acts = gpu_from_host(acts)  # this should get optimized away
    return GpuCtc()(acts, labels, input_lengths)
  else:
    return CpuCtc()(acts, labels, input_lengths)

