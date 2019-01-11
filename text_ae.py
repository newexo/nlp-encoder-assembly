from keras.layers import Input
from keras.models import Model

import hyper_params as hp


class TrainingHyper(hp.Hyper):
    def __init__(self,
            lr=0.001,
            batch_size=32,
            epochs=5,
            max_len=64):
        hp.Hyper.__init__(self, lr, batch_size, epochs)
        self.max_len = max_len

    def display(self):
        hp.Hyper.display(self)
        print("max length=%d" % self.max_len)


class TextAE(object):
    def __init__(self, encoder, decoder):
        self.encoder = encoder
        self.decoder = decoder

    def make_model(self, hyper):
        x = Input(shape=(hyper.max_len,), name='text_input')
        h = self.encoder(x)
        h = self.decoder(h, hyper.max_len)
        model = Model(x, h)
        model.compile(optimizer=hyper.make_optimizer(),
                      loss='categorical_crossentropy',
                      metrics=['categorical_accuracy'])

        return model


class TextAEHyper(object):
    def __init__(self, encoder, decoder, trainer):
        self.encoder = encoder
        self.decoder = decoder
        self.trainer = trainer

    def display(self):
        self.encoder.display()
        self.decoder.display()
        self.trainer.display()

    def make_layer(self):
        encoder = self.encoder.make_layer()
        decoder = self.decoder.make_layer()
        return TextAE(encoder, decoder)
