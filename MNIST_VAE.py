from keras.layers import Input, Dense, Lambda, Layer
from keras.models import Model
from keras.optimizers import RMSprop
from keras import backend as K
from keras import metrics


class Hyper(object):
    def __init__(self,
            batch_size = 100,
            lr=0.001, 
            original_dim = 784,
            latent_dim = 2,
            intermediate_dim = 256,
            epochs = 50,
            epsilon_std = 1.0):
        self.batch_size = batch_size
        self.lr = lr
        self.original_dim = original_dim
        self.latent_dim = latent_dim
        self.intermediate_dim = intermediate_dim
        self.epochs = epochs
        self.epsilon_std = epsilon_std


class MnistVae(object):
    def __init__(self, hyper):
        self.h = hyper

        x, h = self.build_encoder()
        self.vae = self.build_vae(x, h)
        self.generator = self.build_generator()

    def build_optimizer(self):
        return RMSprop(lr=self.h.lr)
        
    def build_encoder(self):
        x = Input(batch_shape=(None, self.h.original_dim))
        return x, Dense(self.h.intermediate_dim, activation='relu')(x)
    
    def build_sampler(self):
        # it seems that sampling function cannot have references to objects like self.h
        latent_dim = self.h.latent_dim
        epsilon_std = self.h.epsilon_std
        
        def sampling(args):
            z_mean, z_log_var = args
            batch_size = K.shape(z_mean)[0]
            epsilon = K.random_normal(shape=(batch_size, latent_dim), mean=0,
                                      stddev=epsilon_std)
            return z_mean + K.exp(z_log_var / 2) * epsilon
        
        return sampling
    
    def build_vae(self, x, h):
        z_mean = Dense(self.h.latent_dim)(h)
        z_log_var = Dense(self.h.latent_dim)(h)
        sampling = self.build_sampler()
        z = Lambda(sampling, output_shape=(self.h.latent_dim,))([z_mean, z_log_var])

        self.decoder_layers = self.build_decoder_layers()
        x_decoded_mean = self.build_decoder(z)

        def zero_loss(y_true, y_pred):
            return K.zeros_like(y_pred)

        original_dim = self.h.original_dim

        class VariationalLossLayer(Layer):
            def __init__(self, **kwargs):
                self.is_placeholder = True
                super(VariationalLossLayer, self).__init__(**kwargs)

            def vae_loss(self, x, x_decoded_mean):
                xent_loss = original_dim * metrics.binary_crossentropy(x, x_decoded_mean)
                kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
                return K.mean(xent_loss + kl_loss)

            def call(self, inputs):
                x, x_decoded_mean = inputs
                loss = self.vae_loss(x, x_decoded_mean)
                self.add_loss(loss, inputs=inputs)

                return K.ones_like(x)

        loss_layer = VariationalLossLayer()([x, x_decoded_mean])
        vae = Model(x, [loss_layer])
        optimizer = self.build_optimizer()
        vae.compile(optimizer=optimizer, loss=[zero_loss])
        
        self.encoder = Model(x, z_mean)

        return vae
        
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
    
    def get_weights(self):
        return self.vae.get_weights()

    def set_weights(self, weights):
        return self.vae.set_weights(weights)

    def fit(self, *args, **kwargs):
        return self.vae.fit(*args, **kwargs)
        
    def save(self, *args, **kwargs):
        self.vae.save(*args, **kwargs)

    def load_weights(self, *args, **kwargs):
        self.vae.load_weights(*args, **kwargs)
        
    def encode(self, x):
        return self.encoder.predict(x)
    
    def generate(self, z):
        return self.generator.predict(z)
