import seq_transform


class TextEncoder(seq_transform.SeqTransform):
    def __init__(self, embedder, layers):
        seq_transform.SeqTransform.__init__(self, layers)
        self.embedder = embedder

    def __call__(self, x):
        h = self.embedder(x)
        return seq_transform.SeqTransform.__call__(self, h)


class Hyper(seq_transform.Hyper):
    def __init__(self, embedder, layers):
        seq_transform.Hyper.__init__(self, layers)
        self.embedder = embedder

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
        return self.transform

    def display(self):
        self.embedder.display()
        print()
        seq_transform.Hyper.display(self)
    
    def make_layer(self, name='encoder'):
        embedder = self.embedder.make_layer()
        layers = seq_transform.Hyper.make_layer(self, name)
        return TextEncoder(embedder, layers.layers)
