import tensorflow as tf
from tensorflow import Tensor
from typing import List


IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_CHANNELS = 3

STEPS = {0: 0.125, 1: 0.25, 2: 0.5, 1: 1}


class UNet:
    @classmethod
    def contraction_path(cls, inputs, step) -> List[Tensor]:
        convolution = tf.keras.layers.Conv2D(
            IMG_WIDTH * step,
            (3, 3),
            activation="relu",
            kernel_initializer="he_normal",
            padding="same",
        )(inputs)
        convolution = tf.keras.layers.Dropout(0.1)(convolution)
        convolution = tf.keras.layers.Conv2D(
            IMG_WIDTH * step,
            (3, 3),
            activation="relu",
            kernel_initializer="he_normal",
            padding="same",
            name=f"contraction_path-{step}"
        )(convolution)
        return [convolution, tf.keras.layers.MaxPooling2D((2, 2))(convolution)]  # type: ignore

    @classmethod
    def expansion_path(cls, inputs: Tensor, joinTo, filters, axis=-1) -> tf.keras.layers.Conv2D:
        convolution = tf.keras.layers.Conv2DTranspose(
            filters, (2, 2), strides=(2, 2), padding="same"
        )(inputs)
        convolution = tf.keras.layers.concatenate([convolution, joinTo], axis)
        convolution = tf.keras.layers.Conv2D(
            IMG_WIDTH,
            (3, 3),
            activation="relu",
            kernel_initializer="he_normal",
            padding="same",
        )(convolution)
        convolution = tf.keras.layers.Dropout(0.2)(convolution)
        return tf.keras.layers.Conv2D(
            IMG_WIDTH,
            (3, 3),
            activation="relu",
            kernel_initializer="he_normal",
            padding="same",
            name=f"expansion_path-{filters}"
        )(
            convolution
        )  # type: ignore

    @classmethod
    def generate_paths(cls, encoded_shape) -> Tensor:
        c1, p1 = UNet.contraction_path(encoded_shape, 0.125)
        c2, p2 = UNet.contraction_path(p1, 0.25)
        c3, p3 = UNet.contraction_path(p2, 0.5)
        c4, p4 = UNet.contraction_path(p3, 1)
        c5, _ = UNet.contraction_path(p4, 2)
        c6 = UNet.expansion_path(c5, c4, 128)
        c7 = UNet.expansion_path(c6, c3, 64)
        c8 = UNet.expansion_path(c7, c2, 32)
        return UNet.expansion_path(c8, c1, 16, axis=3)

    def __init__(
        self,
        input_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS),
        num_classes=1,
        encoder_config=None,
    ):
        # Build the model
        inputs = tf.keras.layers.Input(input_shape)
        s = tf.keras.layers.Lambda(lambda x: x / 255)(inputs)

        c9 = UNet.generate_paths(s)

        outputs = tf.keras.layers.Conv2D(1, (1, 1), activation="sigmoid")(c9)

        self.model = tf.keras.Model(inputs=[inputs], outputs=[outputs])

    def compile(self, optimizer, loss, metrics):
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    def summary(self):
        return self.model.summary()
