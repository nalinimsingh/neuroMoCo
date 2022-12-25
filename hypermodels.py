import math
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import *


def get_motion_hypernetwork(input_shape = [18], n_layers=6, n_units=256):
    """Constructs a hypernetwork and associated network.

    The hypernetwork has eighteen input parameters: For each of 6 shots, a
    horizontal and vertical translation, and a counterclockwise rotation.  """

    hyp_input = tf.keras.Input(shape=input_shape, name='hyp_input')
    hyp_last = hyp_input

    for n in range(n_layers):
        if(n==n_layers-1):
            n_units = 16
        hyp_last = Dense(n_units, activation='relu',
                         name='hyp_dense_%d' % (n + 1),
                         bias_initializer = "glorot_uniform")(hyp_last)

    hyp_model = tf.keras.Model(inputs=hyp_input, outputs=hyp_last,
                               name='hypernet')

    return hyp_model
