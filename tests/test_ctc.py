# This test is designed to mirror the tutorial in Torch, found here:
#   https://github.com/baidu-research/warp-ctc/blob/master/torch_binding/TUTORIAL.md

import numpy as np
import theano
import theano.tensor as T

from theano_ctc import ctc_cost

from theano.printing import debugprint as dprint

def softmax(x):
  """Compute softmax values for each sets of scores in x."""
  broadcastShape = x.shape[0:2] + (1,)
  e_x = np.exp(x - np.max(x, axis=2).reshape(broadcastShape))
  return e_x / e_x.sum(axis=2).reshape(broadcastShape)

targetN = 5
batchSize = 1
seqLen = 1

# time, batchSize, targetN
acts = np.asarray([[[0,0,0,0,0],[1,2,3,4,5],[-5,-4,-3,-2,-1]],\
                   [[0,0,0,0,0],[6,7,8,9,10],[-10,-9,-8,-7,-6]],
                   [[0,0,0,0,0],[11,12,13,14,15], [-15,-14,-13,-12,-11]]], dtype=np.float32)

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

tsActs = theano.shared(acts, "acts")   # float32, so should be moved to GPU
tsActT = theano.shared(actT, "actT")
tsLabels = theano.shared(labels, "labels")
tsLabelT = theano.shared(labelT, "labelT")

# CTC cost
tCost = ctc_cost(tsActs, tsActT, tsLabels, tsLabelT)

# Gradient of CTC cost
tGrad = T.grad(T.mean(tCost), tsActs)

f = theano.function([], [tCost, tGrad])

print "Theano function to calculate costs and gradient of mean(costs):"
dprint(f)
print "\n"

cost, grad = f()
print "cost:"
print cost
print
print "gradient:"
print np.asarray(grad)
