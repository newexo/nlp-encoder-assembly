# Construct a model with same architecture as from Alex Adam
# http://alexadam.ca/ml/2017/05/05/keras-vae.html

from keras.layers import Dense

from nlp_enc import vae_old as vae


class VAEAlexAdam(vae.vae):
    def __init__(self, h):
        vae.vae.__init__(self, h)

    def build_auxiliary(self, encoded):
        h = Dense(100, activation='linear')(encoded)

        return Dense(1, activation='sigmoid', name='pred')(h)
