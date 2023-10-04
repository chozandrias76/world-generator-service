import unittest
from unittest.mock import patch
from src.unet import UNet
import tensorflow as tf


class TestUNet(unittest.TestCase):
    def setUp(self):
        self.unet_instance = UNet()

    def test_model(self):
        unet_model_type = type(self.unet_instance.model)
        tf_keras_model_parent = tf.keras.Model.__bases__[0]
        self.assertTrue(issubclass(unet_model_type, tf_keras_model_parent))


class TestUNetModelLayers(unittest.TestCase):
    def setUp(self):
        self.unet_instance = UNet()  # Initialize an instance of your UNet class

    def test_layers(self):
        # Get the model from your UNet instance
        model = self.unet_instance.model

        # List to hold the types of layers you expect in sequence
        expected_build_layers = [tf.keras.layers.InputLayer, tf.keras.layers.Lambda]
        expected_c1_path_layers = [
            tf.keras.layers.Conv2D,
            tf.keras.layers.Dropout,
            tf.keras.layers.Conv2D,
            tf.keras.layers.MaxPooling2D,
        ]
        expected_c2_path_layers = list(expected_c1_path_layers)
        expected_c3_path_layers = list(expected_c1_path_layers)
        expected_c4_path_layers = list(expected_c1_path_layers)
        expected_c5_path_layers = [
            tf.keras.layers.Conv2D,
            tf.keras.layers.Dropout,
            tf.keras.layers.Conv2D,
        ]
        expected_contraction_path = (
            expected_c1_path_layers
            + expected_c2_path_layers
            + expected_c3_path_layers
            + expected_c4_path_layers
            + expected_c5_path_layers
        )

        expected_c6_path = [
            tf.keras.layers.Conv2DTranspose,
            tf.keras.layers.concatenate,
            tf.keras.layers.Conv2D,
            tf.keras.layers.Dropout,
            tf.keras.layers.Conv2D
        ]
        expected_c7_path = list(expected_c6_path)
        expected_c8_path = list(expected_c6_path)
        expected_c9_path = list(expected_c6_path)
        expected_expansion_path = (
            expected_c6_path
            + expected_c7_path
            + expected_c8_path
            + expected_c9_path
        )
        expected_output_paths = [
            tf.keras.layers.Conv2D
        ]

        expected_layers = (
            expected_build_layers
            + expected_contraction_path
            + expected_expansion_path
            + expected_output_paths
        )

        # Check if the layers in the model are of the same type as in expected_layers
        for layer, expected_layer in zip(model.layers, expected_layers):
            try:
                self.assertTrue(isinstance(layer, expected_layer), msg=f"layer {layer} should be an instance of {expected_layer}")
            except TypeError:
                # concatenate
                self.assertTrue(
                    callable(layer)
                )


if __name__ == "__main__":
    unittest.main()
