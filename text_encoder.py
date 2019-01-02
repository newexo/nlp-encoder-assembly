import hyper_params

class Hyper(vae.Hyper):
    def __init__(self, embedder, layers):
        self.embedder = embedder
        self.layers = layers

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
    