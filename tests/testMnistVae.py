import unittest
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from pathlib import Path

import numpy as np
from numpy.linalg import norm
import tensorflow as tf

from keras.backend import backend as K
from numpy.random import seed

import vae
from MNIST_VAE import Hyper, MnistVae
from . import random_init

class TestMnistVae(unittest.TestCase):
    @property
    def file_path(self):
        return Path('models/test_mnist_vae.h5')
    
    def setUp(self):
        self.x = [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.32941177, 0.7254902, 0.62352943, 0.5921569, 0.23529412, 0.14117648, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.87058824, 0.99607843, 0.99607843, 0.99607843, 0.99607843, 0.94509804, 0.7764706, 0.7764706, 0.7764706, 0.7764706, 0.7764706, 0.7764706, 0.7764706, 0.7764706, 0.6666667, 0.20392157, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2627451, 0.44705883, 0.28235295, 0.44705883, 0.6392157, 0.8901961, 0.99607843, 0.88235295, 0.99607843, 0.99607843, 0.99607843, 0.98039216, 0.8980392, 0.99607843, 0.99607843, 0.54901963, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.06666667, 0.25882354, 0.05490196, 0.2627451, 0.2627451, 0.2627451, 0.23137255, 0.08235294, 0.9254902, 0.99607843, 0.41568628, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3254902, 0.99215686, 0.81960785, 0.07058824, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.08627451, 0.9137255, 1.0, 0.3254902, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5058824, 0.99607843, 0.93333334, 0.17254902, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.23137255, 0.9764706, 0.99607843, 0.24313726, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.52156866, 0.99607843, 0.73333335, 0.019607844, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.03529412, 0.8039216, 0.972549, 0.22745098, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.49411765, 0.99607843, 0.7137255, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.29411766, 0.9843137, 0.9411765, 0.22352941, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.07450981, 0.8666667, 0.99607843, 0.6509804, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.011764706, 0.79607844, 0.99607843, 0.85882354, 0.13725491, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.14901961, 0.99607843, 0.99607843, 0.3019608, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.12156863, 0.8784314, 0.99607843, 0.4509804, 0.003921569, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.52156866, 0.99607843, 0.99607843, 0.20392157, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.23921569, 0.9490196, 0.99607843, 0.99607843, 0.20392157, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.4745098, 0.99607843, 0.99607843, 0.85882354, 0.15686275, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.4745098, 0.99607843, 0.8117647, 0.07058824, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.45490196, 0.49019608, 0.67058825, 1.0, 1.0, 0.5882353, 0.3647059, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.6627451, 0.99215686, 0.99215686, 0.99215686, 0.99215686, 0.99215686, 0.99215686, 0.85490197, 0.11764706, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.6627451, 0.99215686, 0.99215686, 0.99215686, 0.8352941, 0.5568628, 0.6901961, 0.99215686, 0.99215686, 0.47843137, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.20392157, 0.98039216, 0.99215686, 0.8235294, 0.1254902, 0.047058824, 0.0, 0.023529412, 0.80784315, 0.99215686, 0.54901963, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3019608, 0.9843137, 0.8235294, 0.09803922, 0.0, 0.0, 0.0, 0.47843137, 0.972549, 0.99215686, 0.25490198, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.12156863, 0.07058824, 0.0, 0.0, 0.0, 0.0, 0.81960785, 0.99215686, 0.99215686, 0.25490198, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.45882353, 0.96862745, 0.99215686, 0.7764706, 0.039215688, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.29803923, 0.96862745, 0.99215686, 0.90588236, 0.24705882, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5019608, 0.99215686, 0.99215686, 0.5647059, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.6901961, 0.9647059, 0.99215686, 0.62352943, 0.047058824, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.09803922, 0.91764706, 0.99215686, 0.9137255, 0.13725491, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.7764706, 0.99215686, 0.99215686, 0.5529412, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.30588236, 0.972549, 0.99215686, 0.7411765, 0.047058824, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.07450981, 0.78431374, 0.99215686, 0.99215686, 0.5529412, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5254902, 0.99215686, 0.99215686, 0.6784314, 0.047058824, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.972549, 0.99215686, 0.99215686, 0.09803922, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.972549, 0.99215686, 0.99215686, 0.16862746, 0.078431375, 0.078431375, 0.078431375, 0.078431375, 0.019607844, 0.0, 0.019607844, 0.078431375, 0.078431375, 0.14509805, 0.5882353, 0.5882353, 0.5882353, 0.5764706, 0.039215688, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.972549, 0.99215686, 0.99215686, 0.99215686, 0.99215686, 0.99215686, 0.99215686, 0.99215686, 0.65882355, 0.56078434, 0.6509804, 0.99215686, 0.99215686, 0.99215686, 0.99215686, 0.99215686, 0.99215686, 0.99215686, 0.48235294, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.68235296, 0.99215686, 0.99215686, 0.99215686, 0.99215686, 0.99215686, 0.99215686, 0.99215686, 0.99215686, 0.99215686, 0.99215686, 0.99215686, 0.9764706, 0.96862745, 0.96862745, 0.6627451, 0.45882353, 0.45882353, 0.22352941, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.4627451, 0.48235294, 0.48235294, 0.48235294, 0.6509804, 0.99215686, 0.99215686, 0.99215686, 0.60784316, 0.48235294, 0.48235294, 0.16078432, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]

        self.expected_shapes = [(784, 256), (256,), (256, 2), (2,), (256, 2), (2,), (2, 256), (256,), (256, 784), (784,)]

        r = np.random.RandomState(42)
        self.expected_weights = random_init.random_from_shapes(r, self.expected_shapes)

        if self.file_path.exists():
            self.file_path.unlink()

        tf.set_random_seed(42)
        session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
        sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
        np.random.seed(42)


    def tearDown(self):
        if self.file_path.exists():
            self.file_path.unlink()

    def test_hyper(self):
        # test default values of hyper parameter object
        hyper = Hyper()

        self.assertEqual(100, hyper.batch_size)
        self.assertEqual(0.001, hyper.lr)
        self.assertEqual(784, hyper.original_dim)
        self.assertEqual(2, hyper.latent_dim)
        self.assertEqual(256, hyper.intermediate_dim)
        self.assertEqual(50, hyper.epochs)
        self.assertEqual(1, hyper.epsilon_std)

    def test_create_vae(self):
        # check that nothing crashes when creating vae
        hyper = Hyper()
        model = MnistVae(hyper)

        # check that model weights match weights in vae
        actual = model.get_weights()
        expected = model.vae.get_weights()
        self.assertEqual(len(expected), len(actual))
        for i in range(len(expected)):
            self.assertEqual(0, norm(expected[i] - actual[i]))

        # check that encoder weights match appropriate weights in vae
        actual = model.encoder.get_weights()
        expected = model.get_weights()[:4]
        self.assertEqual(len(expected), len(actual))
        for i in range(len(expected)):
            self.assertEqual(0, norm(expected[i] - actual[i]))

        # check that generator weights match appropriate weights in vae
        actual = model.generator.get_weights()
        expected = model.get_weights()[-4:]
        self.assertEqual(len(expected), len(actual))
        for i in range(len(expected)):
            self.assertEqual(0, norm(expected[i] - actual[i]))

    def test_shapes(self):
        # test that weights are correctly shaped in vae
        hyper = Hyper()
        model = MnistVae(hyper)

        actual = random_init.get_shapes(model)
        expected = self.expected_shapes

        self.assertEqual(expected, actual)

        # check that encoder weights shapes match appropriate weight shapes in vae
        actual = random_init.get_shapes(model.encoder)
        expected = self.expected_shapes[:4]
        self.assertEqual(expected, actual)

        # check that generator weights shapes match appropriate weight shapes in vae
        actual = random_init.get_shapes(model.generator)
        expected = self.expected_shapes[-4:]
        self.assertEqual(expected, actual)
        
    def test_random_init(self):
        hyper = Hyper()
        model = MnistVae(hyper)

        r = np.random.RandomState(42)
        model.set_weights(random_init.random_w(r, model))

        # test that model weights are actually as expected
        actual = model.get_weights()
        expected = self.expected_weights
        self.assertEqual(len(expected), len(actual))
        for i in range(len(expected)):
            self.assertAlmostEqual(0, norm(expected[i] - actual[i]), 6)

        # check that model weights match weights in vae
        actual = model.vae.get_weights()
        expected = self.expected_weights
        self.assertEqual(len(expected), len(actual))
        for i in range(len(expected)):
            self.assertAlmostEqual(0, norm(expected[i] - actual[i]), 6)

        # check that encoder weights match appropriate weights in vae
        actual = model.encoder.get_weights()
        expected = self.expected_weights[:4]
        self.assertEqual(len(expected), len(actual))
        for i in range(len(expected)):
            self.assertAlmostEqual(0, norm(expected[i] - actual[i]), 6)

        # check that generator weights match appropriate weights in vae
        actual = model.generator.get_weights()
        expected =self.expected_weights[-4:]
        self.assertEqual(len(expected), len(actual))
        for i in range(len(expected)):
            self.assertAlmostEqual(0, norm(expected[i] - actual[i]), 6)

    def test_save_and_load(self):
        hyper = Hyper()
        model0 = MnistVae(hyper)
        model = MnistVae(hyper)

        r = np.random.RandomState(42)
        model0.set_weights(random_init.random_w(r, model))

        model0.save(self.file_path)
        model.load_weights(self.file_path)

        # test that model weights are actually as expected
        actual = model.get_weights()
        expected = self.expected_weights
        self.assertEqual(len(expected), len(actual))
        for i in range(len(expected)):
            self.assertAlmostEqual(0, norm(expected[i] - actual[i]), 6)

        # check that model weights match weights in vae
        actual = model.vae.get_weights()
        expected = self.expected_weights
        self.assertEqual(len(expected), len(actual))
        for i in range(len(expected)):
            self.assertAlmostEqual(0, norm(expected[i] - actual[i]), 6)

        # check that encoder weights match appropriate weights in vae
        actual = model.encoder.get_weights()
        expected = self.expected_weights[:4]
        self.assertEqual(len(expected), len(actual))
        for i in range(len(expected)):
            self.assertAlmostEqual(0, norm(expected[i] - actual[i]), 6)

        # check that generator weights match appropriate weights in vae
        actual = model.generator.get_weights()
        expected =self.expected_weights[-4:]
        self.assertEqual(len(expected), len(actual))
        for i in range(len(expected)):
            self.assertAlmostEqual(0, norm(expected[i] - actual[i]), 6)

    def test_fit(self):
        r = np.random.RandomState(42)
        x_train = r.rand(1000, 784)
        x_test = r.rand(300, 784)

        hyper = Hyper(epochs=2)
        model = MnistVae(hyper)

        history = model.fit(x_train, x_train,
                shuffle=True,
                epochs=hyper.epochs,
                batch_size=hyper.batch_size,
                validation_data=(x_test, x_test))

        actual = repr(history.history)
        expected = "{'val_loss': [543.7113850911459, 543.6832071940104], 'loss': [551.5591918945313, 543.6121643066406]}"
        self.assertEqual(expected, actual)

    def test_model_encode(self):
        r = np.random.RandomState(42)
        hyper = Hyper()
        model = MnistVae(hyper)
        model.set_weights(random_init.random_w(r, model))
        x = r.rand(3, 784)

        actual = model.encoder.predict(x)

        expected = np.array([[8.137277 , 8.068749 ],
                            [8.201659 , 8.117662 ],
                            [7.9859567, 7.910638 ]], dtype=np.float32)

        self.assertAlmostEqual(0, norm(expected - actual))

    def test_model_generate(self):
        r = np.random.RandomState(42)
        hyper = Hyper()
        model = MnistVae(hyper)
        model.set_weights(random_init.random_w(r, model))
        x = r.rand(2, 2)

        actual = model.generator.predict(x)[:, :3]
        expected = np.array([[0.51340175, 0.5130639 , 0.5114397 ],
                            [0.51576674, 0.51533264, 0.51359564]], dtype=np.float32)

        self.assertAlmostEqual(0, norm(expected - actual))
