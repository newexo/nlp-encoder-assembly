from keras.layers import Dense, RepeatVector, TimeDistributed

import seq_transform


class TextDecoder(seq_transform.SeqTransform):
    def __init__(self, layers, dense, upsample):
        seq_transform.SeqTransform.__init__(self, layers)
        self.dense = dense
        self.upsample = upsample

    def __call__(self, x, max_len):
        h = RepeatVector(int(max_len / self.upsample))(x)
        h = seq_transform.SeqTransform.__call__(self, h)
        return TimeDistributed(self.dense)(h)


class Hyper(seq_transform.Hyper):
    def __init__(self, vocab_size, layers):
        seq_transform.Hyper.__init__(self, layers)
        self.vocab_size = vocab_size

    @property
    def default_name(self):
        return 'decoder'
    
    def make_layer(self, name='decoder'):
        dense = Dense(self.vocab_size, activation='softmax', name='probs')
        layers = seq_transform.Hyper.make_layer(self, name)
        return TextDecoder(layers.layers, dense, self.upsample)
