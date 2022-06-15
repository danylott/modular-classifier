import tensorflow as tf
from .pspnet import PSPNet


def builder(num_classes, input_size=(256, 256), base_model=None):
    layers = tf.keras.layers

    net = PSPNet(num_classes, 'PSPNet', base_model)

    inputs = layers.Input(shape=input_size+(3,))

    return net(inputs), net.get_base_model()
