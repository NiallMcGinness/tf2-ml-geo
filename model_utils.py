import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.initializers import he_normal

def create_model():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(3, 3, strides=1, padding="same", use_bias=False, kernel_initializer= he_normal(seed=None) ) )
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPool2D(pool_size=3, strides=2, padding="valid") )
    model.add(layers.LeakyReLU(0.2) )
    model.add(layers.Flatten())
    model.add(layers.Dense(64))
    model.add(layers.LeakyReLU(0.2))
    model.add(layers.Dense(2, activation="relu"))
    return model