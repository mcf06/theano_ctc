import numpy as np
import theano
import theano.tensor as T
from theano.gof import local_optimizer
from theano.tensor.opt import register_canonicalize
from theano.tensor.opt import register_stabilize
from ctc_base import CtcBase

class CpuCtc(CtcBase):
  def createOp(self):
    # Normally a singleton Op instance is created, and different Apply nodes are
    # created for different inputs.
    # Here, we create an Op instance specifically for this application,
    # and store the gradient variable in it so that it can be used by grad().
    op = CpuCtc()
    op.costs = T.fvector(name="ctc_cost")
    op.gradients = T.ftensor3(name="ctc_grad")
    return op

  def make_node(self, acts, labels, input_lengths = None):
    acts = T.as_tensor_variable(acts)
    labels = T.as_tensor_variable(labels)
    if input_lengths != None:
      input_lengths = T.as_tensor_variable(input_lengths)

    return CtcBase.make_node(self, acts, labels, input_lengths)      

  """
   Compute the connectionist temporal classification loss between a sequence
   of probabilities and a ground truth labeling.  Optionally compute the
   gradient with respect to the inputs.
  \param [in] activations pointer to the activations in either CPU or GPU
              addressable memory, depending on info.  We assume a fixed
              memory layout for this 3 dimensional tensor, which has dimension
              (t, n, p), where t is the time index, n is the minibatch index,
              and p indexes over probabilities of each symbol in the alphabet.
              The memory layout is (t, n, p) in C order (slowest to fastest changing
              index, aka row-major), or (p, n, t) in Fortran order (fastest to slowest
              changing index, aka column-major). We also assume strides are equal to
              dimensions - there is no padding between dimensions.
              More precisely, element (t, n, p), for a problem with mini_batch examples
              in the mini batch, and alphabet_size symbols in the alphabet, is located at:
              activations[(t * mini_batch + n) * alphabet_size + p]
  \param [out] gradients if not NULL, then gradients are computed.  Should be
               allocated in the same memory space as probs and memory
               ordering is identical.
  \param [in]  flat_labels Always in CPU memory.  A concatenation
               of all the labels for the minibatch.
  \param [in]  label_lengths Always in CPU memory. The length of each label
               for each example in the minibatch.
  \param [in]  input_lengths Always in CPU memory.  The number of time steps
               for each sequence in the minibatch.
  \param [in]  alphabet_size The number of possible output symbols.  There
               should be this many probabilities for each time step.
  \param [in]  mini_batch How many examples in a minibatch.
  \param [out] costs Always in CPU memory.  The cost of each example in the
               minibatch.
  \param [in,out] workspace In same memory space as probs. Should be of
                  size requested by get_workspace_size.
  \param [in]  ctcComputeInfo describes whether or not the execution should
               take place on the CPU or GPU, and by extension the location of
               the probs and grads pointers.  Can be used to set the
               number of threads for cpu execution or the stream for gpu
               execution.
 
   \return Status information
 
  ctcStatus_t compute_ctc_loss(const float* const activations,
                               float* gradients,
                               const int* const flat_labels,
                               const int* const label_lengths,
                               const int* const input_lengths,
                               int alphabet_size,
                               int minibatch,
                               float *costs,
                               void *workspace,
                               ctcComputeInfo info);

  """

  def c_code(self, node, name, inNames, outNames, sub):
    fail = sub['fail']
    acts = inNames[0]
    input_lengths = inNames[1]
    labels = inNames[2]
    computeGradient = inNames[3]
   
    costs = outNames[0]
    gradients = outNames[1]

    return """

ctcComputeInfo computeInfo;
computeInfo.loc = CTC_CPU;
computeInfo.num_threads = 1;

// INPUTS -----------

float * acts = (dtype_%(acts)s *) PyArray_DATA(%(acts)s); 

SmartPtr<int*> flat_labels;
SmartPtr<int*> label_lengths;
flattenLabels(%(labels)s, flat_labels, label_lengths);

// input_lengths must be <= acts.shape[0]
int * input_lengths = (dtype_%(input_lengths)s *) PyArray_DATA(%(input_lengths)s); 
int minibatch_size = PyArray_DIMS(%(acts)s)[1];
int alphabet_size = PyArray_DIMS(%(acts)s)[2];

// INTERMEDIATES -----------

SmartPtr<void*> ctc_cpu_workspace;

// OUTPUTS -----------

float* costs;
npy_intp cost_size = minibatch_size;
float* gradients = NULL;

int* computeGradientsData = (int*) PyArray_DATA(%(computeGradient)s);
bool computeGradients = (computeGradientsData[0] == 1) ? true : false;

if (%(costs)s == NULL) {
  // Symbolic variable has no real backing, so create one.
  %(costs)s = (PyArrayObject*)PyArray_ZEROS(1, &cost_size, NPY_FLOAT32, 0);
} else if (PyArray_NDIM(%(costs)s) != 1 || PyArray_DIMS(%(costs)s)[0] != cost_size) {
  // Existing matrix is the wrong size. Make a new one.
  // Decrement ref counter to existing array
  Py_XDECREF(%(costs)s); 
  // Allocate new array
  %(costs)s = (PyArrayObject*)PyArray_ZEROS(1, &cost_size, NPY_FLOAT32, 0);
}
if (!%(costs)s)
  %(fail)s;

costs = (dtype_%(costs)s *) PyArray_DATA(%(costs)s);

if (computeGradients) {
  if (%(gradients)s == NULL) {
    // Symbolic variable has no real backing, so create one.
    %(gradients)s = (PyArrayObject*)PyArray_ZEROS(3, PyArray_DIMS(%(acts)s), NPY_FLOAT32, 0);
  } else if (PyArray_NDIM(%(gradients)s) != 3 
             || PyArray_DIMS(%(gradients)s)[0] != PyArray_DIMS(%(acts)s)[0]
             || PyArray_DIMS(%(gradients)s)[1] != PyArray_DIMS(%(acts)s)[1]
             || PyArray_DIMS(%(gradients)s)[2] != PyArray_DIMS(%(acts)s)[2]) {
    // Existing matrix is the wrong size. Make a new one.
    // Decrement ref counter to existing array
    Py_XDECREF(%(gradients)s); 
    // Allocate new array
    %(gradients)s = (PyArrayObject*)PyArray_ZEROS(3, PyArray_DIMS(%(acts)s), NPY_FLOAT32, 0);
  }
  if (!%(gradients)s)
    %(fail)s;

  gradients = (dtype_%(gradients)s *) PyArray_DATA(%(gradients)s);
}

// COMPUTE -----------

ctcStatus_t status;
size_t cpu_workspace_size;
status = 
  get_workspace_size(label_lengths, input_lengths, alphabet_size, minibatch_size, computeInfo,
                     &cpu_workspace_size);

if (status != CTC_STATUS_SUCCESS) {
  std::cout << "warpctc.get_workspace_size() exited with status " << status << std::endl;
  %(fail)s;
}

ctc_cpu_workspace = malloc(cpu_workspace_size);

status = 
  compute_ctc_loss(acts, gradients, flat_labels, label_lengths, input_lengths, alphabet_size,
                   minibatch_size, costs, ctc_cpu_workspace, computeInfo);

if (status != CTC_STATUS_SUCCESS) {
  std::cout << "warpctc.compute_ctc_loss() exited with status " << status << std::endl;
  %(fail)s;
}
    """ % locals()

cpu_ctc_cost = CpuCtc()

# Disable gradient computation if not needed
@register_canonicalize 
@register_stabilize 
@local_optimizer([CpuCtc]) 
def local_CpuCtc_no_grad(node): 
  if isinstance(node.op, CpuCtc): 
    if len(node.outputs[1].clients) == 0: 
      node.op.computeGradient.set_value(np.asarray([0], dtype=np.int32))
