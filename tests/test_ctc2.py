# This test is designed to mirror the tutorial in Torch, found here:
#   https://github.com/baidu-research/warp-ctc/blob/master/torch_binding/TUTORIAL.md
#
# In the tutorial, the activations are directly defined.
# To test evaluation of a more complicated gradient, here we compute the activations
# as a linear transform of inputs. The inputs and weights are constructed so as to
# generate the activations found in the tutorial.
# This test computes the gradient with respect to the weights.

import numpy as np
import theano
import theano.tensor as T

from theano_ctc import ctc_cost

from theano.printing import debugprint as dprint

import sys

def softmax(x):
  """Compute softmax values for each sets of scores in x."""
  broadcastShape = x.shape[0:2] + (1,)
  e_x = np.exp(x - np.max(x, axis=2).reshape(broadcastShape))
  return e_x / e_x.sum(axis=2).reshape(broadcastShape)

targetN = 5
batchSize = 1
seqLen = 1

# time, batchSize, inputLayerSize
inputs = np.asarray([[[0, 0, 0, 0, 0, 0], [1, 2, 3, 4, 5,  0], [1, 2, 3, 4, 5, -6]], \
                     [[0, 0, 0, 0, 0, 0], [1, 2, 3, 4, 5,  5], [1, 2, 3, 4, 5, -11]], \
                     [[0, 0, 0, 0, 0, 0], [1, 2, 3, 4, 5, 10], [1, 2, 3, 4, 5, -16]]], \
                    dtype=np.float32)

# weight matrix: inputLayerSize x outputLayerSize
weights = np.asarray([[1, 0, 0, 0, 0], \
                      [0, 1, 0, 0, 0], \
                      [0, 0, 1, 0, 0], \
                      [0, 0, 0, 1, 0], \
                      [0, 0, 0, 0, 1], \
                      [1, 1, 1, 1, 1]], dtype=np.float32)

# time, batchSize, outputLayerSize
acts = np.dot(inputs, weights)

# actual duration of each sequence
actT = np.asarray([1, 3, 3], dtype=np.int32)

print "Activations"
print acts
print
print "Softmax outputs"
print softmax(acts)
print

# labels for each sequence, flattened
labels = np.asarray([1,  3,3,  2,3], dtype=np.int32)
labelT = np.asarray([1,  2,    2  ], dtype=np.int32)

# Symbolic equivalents
tsInputs = theano.shared(inputs, name="inputs")
tsWeights = theano.shared(weights, name="weights")
tsActs = T.dot(tsInputs, tsWeights)
tsActT = theano.shared(actT, "actT")
tsLabels = theano.shared(labels, "labels")
tsLabelT = theano.shared(labelT, "labelT")

print "tsActs:"
dprint(tsActs)
print

# CTC cost
tCost = ctc_cost(tsActs, tsActT, tsLabels, tsLabelT)

print "Symbolic CTC cost:"
dprint(tCost)
print "\n"

# Gradient of CTC cost
tGrad = T.grad(T.mean(tCost), tsWeights)

print "Symbolic gradient of CTC cost:"
dprint(tGrad)
print "\n"

f = theano.function([], [tCost, tGrad])
print "Theano function to calculate costs and gradient of mean(costs):"
dprint(f)
print

cost, grad = f()
print "cost:"
print cost
print
print "gradient of average ctc_cost with respect to weights:"
print np.asarray(grad)
