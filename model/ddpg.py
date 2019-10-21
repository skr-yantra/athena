import tensorflow as tf

from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, Dense, MaxPool2D, Flatten


class Base(Model):

    def __init__(self):
        super(Base, self).__init__()
        self._conv1 = Conv2D(32, 9)
        self._conv2 = Conv2D(32, 5)
        self._conv4 = Conv2D(64, 1)
        self._max_pool = MaxPool2D()
        self._flatten = Flatten()

    def call(self, x, training=None, mask=None):
        x = self._conv1(x)
        x = self._conv2(x)
        x = self._max_pool(x)
        x = self._conv4(x)
        x = self._flatten(x)

        return x


class Actor(Base):

    def __init__(self):
        super(Actor, self).__init__()
        self._dense1 = Dense(128)
        self._dense2 = Dense(4)

    def call(self, x, training=None, mask=None):
        x = super(Actor, self).call(x, training, mask)
        x = self._dense1(x)
        x = self._dense2(x)

        return x


class Critic(Base):

    def __init__(self):
        super(Critic, self).__init__()
        self._dense1 = Dense(128)
        self._dense2 = Dense(32)
        self._dense3 = None
