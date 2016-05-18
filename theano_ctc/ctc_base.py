import numpy as np
import theano
import theano.tensor as T
from theano.gof import Op
from theano.gradient import grad_undefined

import os

class CtcBase(Op):
  ctcLibDir = os.environ["CTC_LIB"]

  def c_lib_dirs(self):
    return [os.path.join(self.ctcLibDir, "build")]

  def c_libraries(self):
    return ["warpctc"]

  def c_header_dirs(self):
    return [os.path.join(self.ctcLibDir, "include")]

  def c_headers(self):
    return ["<iostream>", "ctc.h"]

  def createOp(self):
    raise Exception("Called createOp() in abstract base class CtcBase")

  def make_node(self, acts, labels, input_lengths = None):
    # Unless specified, assume all sequences have full sequence length, i.e. acts_.shape[0]
    if input_lengths == None:
      input_lengths = T.cast(acts.shape[0], dtype="int32") * T.ones_like(acts[0,:,0], dtype=np.int32)

    # acts.shape = [seqLen, batchN, outputUnit]
    if acts.dtype != "float32":
      raise Exception("acts must be float32 instead of %s" % acts.dtype)
    # labels.shape = [batchN, labelLen]
    if labels.dtype != "int32":
      raise Exception("labels must be int32 instead of %s" % labels.dtype)
    # input_lengths.shape = [batchN]
    if input_lengths.dtype != "int32":
      raise Exception("input_lengths must be int32 instead of %s" % input_lengths.dtype)

    op = self.createOp()

    # Don't compute gradient unless needed
    op.computeGradient = theano.shared(np.asarray([1], dtype=np.int32))

    applyNode = theano.Apply(op, 
                             inputs=[acts, input_lengths, labels, op.computeGradient], 
                             outputs=[op.costs, op.gradients])

    # Return only the cost. Gradient will be returned by grad()
    self.default_output = 0   
    return applyNode

  def grad(self, inputs, output_grads):
    return [self.gradients,
            grad_undefined(self, 1, inputs[1]),
            grad_undefined(self, 2, inputs[2]),
            grad_undefined(self, 3, inputs[3])]

  def c_support_code(self):
    return """

template <typename T, typename FreeFunc = void (*)(void*)>  // A smart pointer for malloc/free
struct SmartPtr {
  T ptr;
  FreeFunc freeFunc_;
  SmartPtr(FreeFunc freeFunc = free) : ptr(0), freeFunc_(freeFunc) {}
  SmartPtr(T p, FreeFunc freeFunc = free) : ptr(p), freeFunc_(freeFunc) {}
  SmartPtr& operator= (T p) { ptr = p; return *this; }
  operator T() { return ptr; }
  ~SmartPtr() { if (ptr != 0)  freeFunc_(ptr); }
};

void flattenLabels(PyArrayObject* labelMatrix, SmartPtr<int*> &flatLabels, SmartPtr<int*>& labelT) {
  int* labels = (int*) PyArray_DATA(labelMatrix);

  int m = PyArray_DIMS(labelMatrix)[0];
  int n = PyArray_DIMS(labelMatrix)[1];

  flatLabels = (int*) malloc( m * n * sizeof(int) );  // allocate max size; okay if not filled
  labelT = (int*) malloc( m * sizeof(int) );

  int f = 0;
  for (int i = 0; i < m; ++i) {
    int count = 0;
    for (int j = 0; j < n; ++j) {
      int label = labels[n * i + j];
      if (label >= 0) {
        flatLabels[f++] = label;
        ++count;
      }
    }
    labelT[i] = count;
  }
}
    """
