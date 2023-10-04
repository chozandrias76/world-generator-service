import unittest
from unittest.mock import patch
from src.unet import UNet
import tensorflow as tf

class TestUNet(unittest.TestCase):
    def setUp(self):
        self.unet_instance  = UNet()

    def test_model(self):
        unet_model_type = type(self.unet_instance.model)
        tf_keras_model_parent = tf.keras.Model.__bases__[0]
        self.assertTrue(issubclass(unet_model_type, tf_keras_model_parent))

if __name__ == "__main__":
    unittest.main()