import hyper_params as hp


class TextEncoder(object):
    def __init__(self, embedder, layers):
        self.embedder = embedder
        self.layers = layers

    def __call__(self, x):
        h = self.embedder(x)
        for layer in self.layers:
            h = layer(h)
        return h


class Hyper(object):
    def __init__(self, embedder, layers):
        self.embedder = embedder
        self.layers = layers

    @property
    def default_name(self):
        return 'encoder'
    
    @property
    def vocab_size(self):
        return self.embedder.vocab_size
    
    @property
    def embedding_dim(self):
        return self.embedder.embedding_dim
    
    @property
    def encoding(self):
        return self.layers[-1]
    
    @property
    def return_sequences(self):
        return self.encoding.return_sequences

    def display(self):
        self.embedder.display()
        for layer in self.layers:
            layer.display()
    
    def make_layer(self, name='encoder'):
        prefix = hp.name_prefix(name)
        embedder = self.embedder.make_layer()
        layers = [layer.make_layer(name=prefix + layer.default_name) for layer in self.layers]
        return TextEncoder(embedder, layers)
