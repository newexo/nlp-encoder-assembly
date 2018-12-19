import numpy as np

from keras import objectives, backend as K
from keras.layers import Input, Dense, Embedding, Bidirectional, LSTM, TimeDistributed, RepeatVector
from keras.models import Model
from keras.optimizers import Adam
from keras.utils import to_categorical

import vae


class Hyper(vae.Hyper):
    def __init__(self,
            vocab_size=1000, 
            embedding_dim=64, 
            max_length=300,
            batch_size=10,
            lr=0.001, 
            latent_dim=435,
            intermediate_dim=200,
            encoder_hidden_dim=500, 
            decoder_hidden_dim=500,
            epochs=50,
            epsilon_std=0.01):
        vae.Hyper.__init__(self,            
            batch_size=batch_size,
            lr=lr, 
            latent_dim=latent_dim,
            intermediate_dim=intermediate_dim,
            epochs=epochs,
            epsilon_std=epsilon_std)
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.max_length = max_length
        self.encoder_hidden_dim=encoder_hidden_dim
        self.decoder_hidden_dim=decoder_hidden_dim


class TextVae(vae.Vae):
    def __init__(self, hyper):
        vae.Vae.__init__(self, hyper)

    def build_optimizer(self):
        return Adam(lr=self.h.lr)
        
    def compute_vae_loss(self, x, x_decoded_mean, z_mean, z_log_var):
        x = K.flatten(x)
        x_decoded_mean = K.flatten(x_decoded_mean)
        xent_loss = self.h.max_length * objectives.binary_crossentropy(x, x_decoded_mean)
        kl_loss = - 0.5 * K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        return xent_loss + kl_loss
        
    def build_encoder(self):
        x = Input(shape=(self.h.max_length,), name='text_input')
        self.embedder = Embedding(self.h.vocab_size, 
            self.h.embedding_dim, 
            input_length=self.h.max_length, name='embedder')
        self.encoder_rnn_1 = Bidirectional(LSTM(self.h.encoder_hidden_dim, 
                return_sequences=True), 
            merge_mode='concat',
            name='encoder_rnn_1')
        self.encoder_rnn_2 = Bidirectional(LSTM(self.h.encoder_hidden_dim, 
                return_sequences=False),
            merge_mode='concat', 
            name='encoder_rnn_2')
        h = self.embedder(x)
        h = self.encoder_rnn_1(h)
        h = self.encoder_rnn_2(h)

        return x, Dense(self.h.intermediate_dim, activation='relu', name='encoder_output')(h)
            
    def build_decoder_layers(self):
        decoder_rnn_1 = LSTM(self.h.decoder_hidden_dim, 
            return_sequences=True, 
            name='decoder_rnn_1')
        decoder_rnn_2 = LSTM(self.h.decoder_hidden_dim,
            return_sequences=True, 
            name='decoder_rnn_2')
        decoder_mean = TimeDistributed(Dense(self.h.vocab_size, activation='softmax'), name='decoded_mean')
        return decoder_rnn_1, decoder_rnn_2, decoder_mean

    def build_decoder(self, z):
        h_decoded = decoder_h(z)
        return decoder_mean(h_decoded)

    def build_decoder(self, encoded):   
        decoder_rnn_1, decoder_rnn_2, decoder_mean = self.decoder_layers

        h = RepeatVector(self.h.max_length)(encoded)
        h = decoder_rnn_1(h)
        h = decoder_rnn_2(h)

        return decoder_mean(h)
    
    def build_generator(self):
        decoder_rnn_1, decoder_rnn_2, decoder_mean = self.decoder_layers

        decoder_input = Input(shape=(self.h.latent_dim,))

        h = RepeatVector(self.h.max_length)(decoder_input)
        h = decoder_rnn_1(h)
        h = decoder_rnn_2(h)

        return Model(decoder_input, decoder_mean(h))
