from keras import objectives, backend as K
from keras.layers import Bidirectional, Dense, Embedding, Input, Lambda, LSTM, RepeatVector, TimeDistributed
from keras.models import Model
import keras

from keras.callbacks import ModelCheckpoint
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import os


class Hyper(object):
    def __init__(self, vocab_size=500, 
                        embedding_dim=64, 
                        max_length=300, 
                        latent_rep_size=200, 
                        encoder_hidden_dim=500, 
                        decoder_hidden_dim=500,
                        encoder_output_dim=435,
                        epsilon_std=0.01):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.max_length = max_length
        self.latent_rep_size = latent_rep_size
        self.encoder_hidden_dim = encoder_hidden_dim
        self.decoder_hidden_dim = decoder_hidden_dim
        self.encoder_output_dim = encoder_output_dim
        self.epsilon_std = epsilon_std


class vae(object):
    def __init__(self, h):
        self.h = h
        self.encoder = None
        self.decoder = None
        self.sentiment_predictor = None
        self.autoencoder = None

        x = Input(shape=(self.h.max_length,))
        x_embed = Embedding(self.h.vocab_size, self.h.embedding_dim, input_length=self.h.max_length)(x)

        vae_loss, encoded = self.build_encoder(x_embed)
        self.encoder = Model(inputs=x, outputs=encoded)

        encoded_input = Input(shape=(self.h.latent_rep_size,))
        predicted_sentiment = self.build_auxiliary(encoded_input)
        self.sentiment_predictor = Model(encoded_input, predicted_sentiment)

        decoded = self.build_decoder(encoded_input)
        self.decoder = Model(encoded_input, decoded)

        self.autoencoder = Model(inputs=x, 
            outputs=[self.build_decoder(encoded), 
            self.build_auxiliary(encoded)])
        self.autoencoder.compile(optimizer='Adam',
                                 loss=[vae_loss, 'binary_crossentropy'],
                                 metrics=['accuracy'])
        
    def build_encoder(self, x):
        h = Bidirectional(LSTM(self.h.encoder_hidden_dim, 
                return_sequences=True, 
                name='encoder_rnn_1'), 
            merge_mode='concat')(x)
        h = Bidirectional(LSTM(self.h.encoder_hidden_dim, 
                return_sequences=False, 
                name='encoder_rnn_2'),
            merge_mode='concat')(h)
        h = Dense(self.h.encoder_output_dim, activation='relu', name='encoder_output')(h)

        latent_rep_size = self.h.latent_rep_size
        epsilon_std = self.h.epsilon_std

        def sampling(args):
            z_mean_, z_log_var_ = args
            batch_size = K.shape(z_mean_)[0]
            epsilon = K.random_normal(shape=(batch_size, 
                    latent_rep_size), 
                mean=0, 
                stddev=epsilon_std)
            return z_mean_ + K.exp(z_log_var_ / 2) * epsilon

        z_mean = Dense(self.h.latent_rep_size, name='z_mean', activation='linear')(h)
        z_log_var = Dense(self.h.latent_rep_size, name='z_log_var', activation='linear')(h)

        def vae_loss(x, x_decoded_mean):
            x = K.flatten(x)
            x_decoded_mean = K.flatten(x_decoded_mean)
            xent_loss = self.h.max_length * objectives.binary_crossentropy(x, x_decoded_mean)
            kl_loss = - 0.5 * K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
            return xent_loss + kl_loss

        return (vae_loss, Lambda(sampling, output_shape=(self.h.latent_rep_size,), name='lambda')([z_mean, z_log_var]))

    def build_decoder(self, encoded):
        repeated_context = RepeatVector(self.h.max_length)(encoded)

        h = LSTM(self.h.decoder_hidden_dim, 
            return_sequences=True, 
            name='decoder_rnn_1')(repeated_context)
        h = LSTM(self.h.decoder_hidden_dim,
            return_sequences=True, 
            name='decoder_rnn_2')(h)

        decoded = TimeDistributed(Dense(self.h.vocab_size, activation='softmax'), name='decoded_mean')(h)

        return decoded
    
    def build_auxiliary(self, encoded):
        # this should break if not overridden
        pass

    def train(self, X_train, X_train_one_hot, y_train, X_test, x_test_one_hot, y_test):
        checkpointer = self.create_model_checkpoint('models', 'rnn_ae')

        print(X_train.shape)
        print(X_train_one_hot.shape)
        print(y_train.shape)

        self.autoencoder.fit(x=X_train, y={'decoded_mean': X_train_one_hot, 'pred': y_train},
                          batch_size=10, epochs=10,
                          validation_data=(X_test, {'decoded_mean': x_test_one_hot, 'pred':  y_test}))
