from __future__ import print_function

import numpy as np
np.random.seed(1337)

import keras.backend as K
from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers.wrappers import TimeDistributed
from keras.layers.recurrent import LSTM
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD

# wrapper of ctc_cost
def ctc_cost(y_true, y_pred):
    '''
    CTC cost:
    a theano wrapper for warp-ctc
    Arguments:
        y_true : label
        y_pred : acts
    '''
    from theano_ctc import ctc_cost as warp_ctc_cost

    # convert (batch size, timestep, target) to (timestep, batch size, target)
    acts = K.permute_dimensions(y_pred, (1, 0, 2))
    labels = K.cast(K.squeeze(y_true, axis=2), 'int32')
    return warp_ctc_cost(acts, labels)

batch_size = 16
frame_len = 80
nb_feat = 120
nb_class = 36
nb_output = nb_class + 1  # add output for blank
inner_dim = 512
nb_cell = 1024

print("Building model...")
model = Sequential()
model.add(LSTM(inner_dim, input_shape = (frame_len, nb_feat), return_sequences = True))
model.add(BatchNormalization())
model.add(TimeDistributed(Dense(nb_output)))

model.summary()

# Compiling
opt = SGD(lr = 1e-5, momentum = 0.9, nesterov = True)
model.compile(optimizer = opt, loss = ctc_cost, sample_weight_mode = None)

# Generate dummy data
data = np.random.uniform(low = -5, high = 5, size = (batch_size, frame_len, nb_feat))
# Dummy labels in range [1,nb_class]. 0 = <blank>
label = 1 + np.random.randint(nb_class, size = (batch_size, frame_len, 1))

# Training
model.fit(data, label, nb_epoch = 5, batch_size = batch_size)
