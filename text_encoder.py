import hyper_params as hp

class Hyper(vae.Hyper):
    def __init__(self, embedder, layers):
        self.embedder = embedder
        self.layers = layers

    @property
    def all_layers(self):
        return [self.embedder] + self.layers
    

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
        return self.encoding._return_sequences

    def display(self):
        self.embedder.display()
        for layer in self.layers:
            layer.display()
    
    def make_layers(self, name='encoder'):
        prefix = hp.name_prefix(name)
        return [layer.make_layer(layer) for layer in self.all_layers]
        