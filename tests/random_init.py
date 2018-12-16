"""
random_init 
Contains functions to discover the shapes of weights in a Keras model 
and provide random weights with which to set the model.

This is mostly for testing purposes.
"""
import math


def scale(shape):
    return 1.0 / math.sqrt(6 * sum(shape))

def get_shapes(model):
    return [w.shape for w in model.get_weights()]

def random_from_shapes(r, shapes):
    return [scale(shape) * r.uniform(size=shape) for shape in shapes]

def random_w(r, model):
    return random_from_shapes(r, get_shapes(model))
