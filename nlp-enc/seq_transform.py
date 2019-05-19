import hyper_params as hp


class SeqTransform(object):
    def __init__(self, layers):
        self.layers = layers

    def __call__(self, x):
        h = x
        for layer in self.layers:
            h = layer(h)
        return h


def product(a):
    p = 1
    for x in a:
        p *= x
    return p


class Hyper(object):
    def __init__(self, layers):
        self.layers = layers

    @property
    def upsample(self):
        return product([layer.upsample for layer in self.layers])
    
    @property
    def downsample(self):
        return product([layer.downsample for layer in self.layers])

    @property
    def transform(self):
        return self.layers[-1]
    
    @property
    def return_sequences(self):
        return self.transform.return_sequences

    def display(self):
        for i, layer in enumerate(self.layers):
            if i:
                print()
            layer.display()
    
    def make_layer(self, name):
        prefix = hp.name_prefix(name)
        layers = [layer.make_layer(name=prefix + layer.default_name) for layer in self.layers]
        return SeqTransform(layers)
