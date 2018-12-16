from keras.layers import Input, Dense, Lambda, Layer
from keras.models import Model
from keras.optimizers import RMSprop
from keras import backend as K
from keras import metrics

import vae


class Hyper(vae.Hyper):
    def __init__(self,
            batch_size=100,
            lr=0.001, 
            original_dim=784,
            latent_dim=2,
            intermediate_dim=256,
            epochs=50,
            epsilon_std=1.0):
        vae.Hyper.__init__(self,            
            batch_size=batch_size,
            lr=lr, 
            original_dim=784,
            latent_dim=latent_dim,
            intermediate_dim=intermediate_dim,
            epochs=epochs,
            epsilon_std=epsilon_std)


class MnistVae(vae.Vae):
    def __init__(self, hyper):
        vae.Vae.__init__(self, hyper)

    def build_optimizer(self):
        return RMSprop(lr=self.h.lr)
        
    def build_encoder(self):
        x = Input(batch_shape=(None, self.h.original_dim))
        return x, Dense(self.h.intermediate_dim, activation='relu')(x)
            
    def build_decoder_layers(self):
        decoder_h = Dense(self.h.intermediate_dim, activation='relu')
        decoder_mean = Dense(self.h.original_dim, activation='sigmoid')
        return decoder_h, decoder_mean

    def build_decoder(self, z):
        decoder_h, decoder_mean = self.decoder_layers
        h_decoded = decoder_h(z)
        return decoder_mean(h_decoded)
    
    def build_generator(self):
        decoder_input = Input(shape=(self.h.latent_dim,))
        decoded_mean = self.build_decoder(decoder_input)
        return Model(decoder_input, decoded_mean)
