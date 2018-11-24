# Modified from Alex Adam
# http://alexadam.ca/ml/2017/05/05/keras-vae.html

from keras import objectives, backend as K
from keras.layers import Bidirectional, Dense, Embedding, Input, Lambda, LSTM, RepeatVector, TimeDistributed
from keras.models import Model
import keras

from keras.callbacks import ModelCheckpoint
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import os

class VAEAlexAdam(object):
    def create(self, vocab_size=500, max_length=300, latent_rep_size=200): # Change max_length to fit phrases
        self.encoder = None
        self.decoder = None
        self.sentiment_predictor = None
        self.autoencoder = None

        x = Input(shape=(max_length,))
        x_embed = Embedding(vocab_size, 64, input_length=max_length)(x)

        vae_loss, encoded = self._build_encoder(x_embed, latent_rep_size=latent_rep_size, max_length=max_length)
        self.encoder = Model(inputs=x, outputs=encoded)

        encoded_input = Input(shape=(latent_rep_size,))
        predicted_sentiment = self._build_sentiment_predictor(encoded_input)
        self.sentiment_predictor = Model(encoded_input, predicted_sentiment)

        decoded = self._build_decoder(encoded_input, vocab_size, max_length)
        self.decoder = Model(encoded_input, decoded)

        self.autoencoder = Model(inputs=x, outputs=[self._build_decoder(encoded, vocab_size, max_length), self._build_sentiment_predictor(encoded)])
        self.autoencoder.compile(optimizer='Adam',
                                 loss=[vae_loss, 'binary_crossentropy'],
                                 metrics=['accuracy'])

        print(self.autoencoder.inputs)
        
    def _build_encoder(self, x, latent_rep_size=200, max_length=300, epsilon_std=0.01):
        h = Bidirectional(LSTM(500, return_sequences=True, name='lstm_1'), merge_mode='concat')(x)
        h = Bidirectional(LSTM(500, return_sequences=False, name='lstm_2'), merge_mode='concat')(h)
        h = Dense(435, activation='relu', name='dense_1')(h)

        def sampling(args):
            z_mean_, z_log_var_ = args
            batch_size = K.shape(z_mean_)[0]
            epsilon = K.random_normal(shape=(batch_size, latent_rep_size), mean=0., stddev=epsilon_std)
            return z_mean_ + K.exp(z_log_var_ / 2) * epsilon

        z_mean = Dense(latent_rep_size, name='z_mean', activation='linear')(h)
        z_log_var = Dense(latent_rep_size, name='z_log_var', activation='linear')(h)

        def vae_loss(x, x_decoded_mean):
            x = K.flatten(x)
            x_decoded_mean = K.flatten(x_decoded_mean)
            xent_loss = max_length * objectives.binary_crossentropy(x, x_decoded_mean)
            kl_loss = - 0.5 * K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
            return xent_loss + kl_loss

        return (vae_loss, Lambda(sampling, output_shape=(latent_rep_size,), name='lambda')([z_mean, z_log_var]))
    
    def _build_decoder(self, encoded, vocab_size, max_length):
        repeated_context = RepeatVector(max_length)(encoded)

        h = LSTM(500, return_sequences=True, name='dec_lstm_1')(repeated_context)
        h = LSTM(500, return_sequences=True, name='dec_lstm_2')(h)

        decoded = TimeDistributed(Dense(vocab_size, activation='softmax'), name='decoded_mean')(h)

        return decoded
    
    def _build_sentiment_predictor(self, encoded):
        h = Dense(100, activation='linear')(encoded)

        return Dense(1, activation='sigmoid', name='pred')(h)

    def create_model_checkpoint(self, dir, model_name):
        filepath = dir + '/' + \
                   model_name + "-{epoch:02d}-{val_decoded_mean_acc:.2f}-{val_pred_loss:.2f}.h5"
        directory = os.path.dirname(filepath)

        try:
            os.stat(directory)
        except:
            os.mkdir(directory)

        checkpointer = ModelCheckpoint(filepath=filepath,
                                       verbose=1,
                                       save_best_only=False)

        return checkpointer

    def train(self, X_train, X_train_one_hot, y_train, X_test, x_test_one_hot, y_test):
        checkpointer = self.create_model_checkpoint('models', 'rnn_ae')

        print(X_train.shape)
        print(X_train_one_hot.shape)
        print(y_train.shape)

        self.autoencoder.fit(x=X_train, y={'decoded_mean': X_train_one_hot, 'pred': y_train},
                          batch_size=10, epochs=10, callbacks=[checkpointer],
                          validation_data=(X_test, {'decoded_mean': x_test_one_hot, 'pred':  y_test}))
