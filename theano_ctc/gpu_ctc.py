import numpy as np
import theano
import theano.tensor as T
from theano.gof import Op
from theano.gof import local_optimizer
from theano.gradient import grad_undefined
from theano.sandbox.cuda import GpuOp
from theano.sandbox.cuda.type import CudaNdarrayType
from theano.sandbox.cuda.var import CudaNdarrayVariable
from theano.tensor.opt import register_canonicalize
from theano.tensor.opt import register_stabilize
import os

class GpuCtc(GpuOp):
  ctcLibDir = os.environ["CTC_LIB"]

  def make_node(self, acts, input_lengths, flat_labels, label_lengths):
    if not isinstance(acts.type, CudaNdarrayType):
      raise Exception("Activations should be CudaNdarrayType, not %s" % (acts.type,))
    acts_ = acts
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
    op = GpuCtc()
    op.costs = T.fvector(name="ctc_cost")
    op.gradients = CudaNdarrayVariable(name="gpu_ctc_grad", 
                                       type=CudaNdarrayType(broadcastable=[False, False, False]))

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
computeInfo.loc = CTC_GPU;
computeInfo.stream = 0;

// INPUTS -----------

float * acts = CudaNdarray_DEV_DATA(%(acts)s); 
int * flat_labels = (dtype_%(flat_labels)s *) PyArray_DATA(%(flat_labels)s); 
int * label_lengths = (dtype_%(label_lengths)s *) PyArray_DATA(%(label_lengths)s); 

// input_lengths must be <= acts.shape[0]
int * input_lengths = (dtype_%(input_lengths)s *) PyArray_DATA(%(input_lengths)s); 
int minibatch_size = CudaNdarray_HOST_DIMS(%(acts)s)[1];
int alphabet_size = CudaNdarray_HOST_DIMS(%(acts)s)[2];

// OUTPUTS -----------

float* costs;
npy_intp costs_size = minibatch_size;  // PyArray_ZEROS wants size as npy_intp[]

float* gradients = NULL;
int gradients_size[3];  // CudaNdarray_ZEROS wants size as int[]
gradients_size[0] = CudaNdarray_HOST_DIMS(%(acts)s)[0];
gradients_size[1] = CudaNdarray_HOST_DIMS(%(acts)s)[1];
gradients_size[2] = CudaNdarray_HOST_DIMS(%(acts)s)[2];

int* computeGradientsData = (int*) PyArray_DATA(%(computeGradient)s);
bool computeGradients = (computeGradientsData[0] == 1) ? true : false;

if (%(costs)s == NULL) {
  // Symbolic variable has no real backing, so create one.
  %(costs)s = (PyArrayObject*)PyArray_ZEROS(1, &costs_size, NPY_FLOAT32, 0);
} else if (PyArray_NDIM(%(costs)s) != 1 || PyArray_DIMS(%(costs)s)[0] != costs_size) {
  // Existing matrix is the wrong size. Make a new one.
  // Decrement ref counter to existing array
  Py_XDECREF(%(costs)s); 
  // Allocate new array
  %(costs)s = (PyArrayObject*)PyArray_ZEROS(1, &costs_size, NPY_FLOAT32, 0);
}
if (!%(costs)s)
  %(fail)s;

costs = (dtype_%(costs)s *) PyArray_DATA(%(costs)s);

if (computeGradients) {
  std::cout << "Computing cost and gradient." << std::endl;

  if (%(gradients)s == NULL) {
    // Symbolic variable has no real backing, so create one.
    %(gradients)s = (CudaNdarray *) CudaNdarray_ZEROS(3, gradients_size);
  } else if (CudaNdarray_NDIM(%(gradients)s) != 3 
             || CudaNdarray_DIMS(%(gradients)s)[0] != gradients_size[0]
             || CudaNdarray_DIMS(%(gradients)s)[1] != gradients_size[1]
             || CudaNdarray_DIMS(%(gradients)s)[2] != gradients_size[2]) {
    // Existing matrix is the wrong size. Make a new one.
    // Decrement ref counter to existing array
    Py_XDECREF(%(gradients)s); 
    // Allocate new array
    %(gradients)s = (CudaNdarray *) CudaNdarray_ZEROS(3, gradients_size);
  }
  if (!%(gradients)s)
    %(fail)s;

  gradients = CudaNdarray_DEV_DATA(%(gradients)s);
}

// COMPUTE -----------

ctcStatus_t status;
void* ctc_gpu_workspace;

size_t gpu_workspace_size;
status = 
  get_workspace_size(label_lengths, input_lengths, alphabet_size, minibatch_size, computeInfo,
                     &gpu_workspace_size);

if (status != CTC_STATUS_SUCCESS) {
  std::cout << "warpctc.get_workspace_size(GPU) exited with status " << status << std::endl;
  %(fail)s;
}

ctc_gpu_workspace = device_malloc(gpu_workspace_size, 1);

status = 
  compute_ctc_loss(acts, gradients, flat_labels, label_lengths, input_lengths, alphabet_size,
                   minibatch_size, costs, ctc_gpu_workspace, computeInfo);

device_free(ctc_gpu_workspace);

if (status != CTC_STATUS_SUCCESS) {
  std::cout << "warpctc.compute_ctc_loss() exited with status " << status << std::endl;
  %(fail)s;
}
    """ % locals()

gpu_ctc_cost = GpuCtc()

# Disable gradient computation if not needed
@register_canonicalize 
@register_stabilize 
@local_optimizer([GpuCtc]) 
def local_GpuCtc_no_grad(node): 
  if not isinstance(node.op, GpuCtc): 
    return 
  if len(node.outputs[1].clients) == 0: 
    node.op.computeGradient.set_value(np.asarray([0], dtype=np.int32))
