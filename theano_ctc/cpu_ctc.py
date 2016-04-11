import numpy as np
import theano
import theano.tensor as T
from theano.gof import Op
from theano.gof import local_optimizer
from theano.gradient import grad_undefined
from theano.tensor.opt import register_canonicalize
from theano.tensor.opt import register_stabilize


import os

class CpuCtc(Op):
  ctcLibDir = os.environ["CTC_LIB"]

  def make_node(self, acts, input_lengths, flat_labels, label_lengths):
    acts_ = T.as_tensor_variable(acts)
    input_lengths_ = T.as_tensor_variable(input_lengths)
    flat_labels_ = T.as_tensor_variable(flat_labels)
    label_lengths_ = T.as_tensor_variable(label_lengths)

    if acts_.dtype != "float32":
      raise Exception("acts must be float32 instead of %s" % acts.dtype)
    if input_lengths.dtype != "int32":
      raise Exception("input_lengths must be int32 instead of %s" % input_lengths.dtype)
    if flat_labels.dtype != "int32":
      raise Exception("flat_labels must be int32 instead of %s" % flat_labels.dtype)
    if label_lengths.dtype != "int32":
      raise Exception("label_lengths must be int32 instead of %s" % label_lengths.dtype)

    # Normally a singleton Op instance is created, and different Apply nodes are
    # created for different inputs.
    # Here, we create an Op instance specifically for this application,
    # and store the gradient variable in it so that it can be used by grad().
    op = CpuCtc()
    op.costs = T.fvector(name="ctc_cost")
    op.gradients = T.ftensor3(name="ctc_grad")

    # Don't compute gradient unless needed
    op.computeGradient = theano.shared(np.asarray([1], dtype=np.int32))

    applyNode = theano.Apply(op, 
                             inputs=[acts_, input_lengths_, flat_labels_, label_lengths_, op.computeGradient], 
                             outputs=[op.costs, op.gradients])

    # Return only the cost. Gradient will be returned by grad()
    self.default_output = 0   
    return applyNode

  def grad(self, inputs, output_grads):
    return [self.gradients,
            grad_undefined(self, 1, inputs[1]),
            grad_undefined(self, 2, inputs[2]),
            grad_undefined(self, 3, inputs[3]),
            grad_undefined(self, 4, inputs[4])]

  def c_lib_dirs(self):
    return [os.path.join(self.ctcLibDir, "build")]

  def c_libraries(self):
    return ["warpctc"]

  def c_header_dirs(self):
    return [os.path.join(self.ctcLibDir, "include")]

  def c_headers(self):
    return ["<iostream>", "ctc.h"]

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
    flat_labels = inNames[2]
    label_lengths = inNames[3] 
    computeGradient = inNames[4]
   
    costs = outNames[0]
    gradients = outNames[1]

    return """

ctcComputeInfo computeInfo;
computeInfo.loc = CTC_CPU;
computeInfo.num_threads = 1;

// INPUTS -----------

float * acts = (dtype_%(acts)s *) PyArray_DATA(%(acts)s); 
int * flat_labels = (dtype_%(flat_labels)s *) PyArray_DATA(%(flat_labels)s); 
int * label_lengths = (dtype_%(label_lengths)s *) PyArray_DATA(%(label_lengths)s); 

// input_lengths must be <= acts.shape[0]
int * input_lengths = (dtype_%(input_lengths)s *) PyArray_DATA(%(input_lengths)s); 
int minibatch_size = PyArray_DIMS(%(acts)s)[1];
int alphabet_size = PyArray_DIMS(%(acts)s)[2];

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
void* ctc_cpu_workspace;

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

free(ctc_cpu_workspace);

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
  if not isinstance(node.op, CpuCtc): 
    return 
  if len(node.outputs[1].clients) == 0: 
    node.op.computeGradient.set_value(np.asarray([0], dtype=np.int32))
