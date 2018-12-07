import math


def scale(shape):
    return 1.0 / math.sqrt(6 * sum(shape))

def random_w(r, model):
    w = model.get_weights()
    return [scale(v.shape) * r.uniform(size=v.shape) for v in w]
