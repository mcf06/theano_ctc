# This test is designed to mirror the tutorial in Torch, found here:
#   https://github.com/baidu-research/warp-ctc/blob/master/torch_binding/TUTORIAL.md

from __future__ import print_function

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

# shape: time, batchSize, targetN
acts = np.asarray([[[0,0,0,0,0],[1,2,3,4,5],[-5,-4,-3,-2,-1]],
                   [[0,0,0,0,0],[6,7,8,9,10],[-10,-9,-8,-7,-6]],
                   [[0,0,0,0,0],[11,12,13,14,15], [-15,-14,-13,-12,-11]]], dtype=np.float32)

# actual duration of each sequence
actT = np.asarray([1, 3, 3], dtype=np.int32)

print("Activations")
print(acts)
print()
print("Softmax outputs")
print(softmax(acts))
print()

# labels for each sequence. -1 = padding to be ignored
labels = np.asarray([[1, -1],
                     [3, 3],
                     [2, 3]], dtype=np.int32)

tsActs = theano.shared(acts, "acts")   # float32, so should be moved to GPU
tsActT = theano.shared(actT, "actT")
tsLabels = theano.shared(labels, "labels")

# CTC cost
tCost = ctc_cost(tsActs, tsLabels, tsActT)

# Gradient of CTC cost
tGrad = T.grad(T.mean(tCost), tsActs)

# Create train (with gradient for SGD) and test (no gradient) functions
train = theano.function([], [tCost, tGrad])
test = theano.function([], [tCost])

print("Theano function to calculate costs and gradient of mean(costs):")
dprint(train)
print("\n")

print("Theano function to calculate costs:")
dprint(test)
print("\n")

cost, grad = train()
print("Training cost:")
print(cost)
print()
print("gradient:")
print(np.asarray(grad))
print("gradient shape:")
print(np.asarray(grad).shape)
print()

cost, = test()
print("Test cost:")
print(cost)
