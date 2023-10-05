import os

import numpy as np
from tqdm import tqdm
from skimage.io import imread, imshow
from skimage.transform import resize


IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_CHANNELS = 3
TRAIN_PATH = "training-data/biology/stage1_train/"
TEST_PATH = "training-data/biology/stage1_test/"


class Canvas:
    def __init__(self):
        self.train_ids = next(os.walk(TRAIN_PATH))[1]
        self.test_ids = next(os.walk(TEST_PATH))[1]
        self.x_train = np.zeros(
            (len(self.train_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8
        )
        self.y_train = np.zeros(
            (len(self.train_ids), IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool_
        )
        self.x_predicts = []

    def resize_images_and_masks(self):
        for n, id_ in tqdm(enumerate(self.train_ids), total=len(self.train_ids)):
            path = TRAIN_PATH + id_
            img = imread(path + "/images/" + id_ + ".png")[:, :, :IMG_CHANNELS]
            img = resize(
                img, (IMG_HEIGHT, IMG_WIDTH), mode="constant", preserve_range=True
            )
            self.x_train[n] = img  # Fill empty X_train with values from img
            mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool_)
            for mask_file in next(os.walk(path + "/masks/"))[2]:
                mask_ = imread(path + "/masks/" + mask_file)
                mask_ = np.expand_dims(
                    resize(
                        mask_,
                        (IMG_HEIGHT, IMG_WIDTH),
                        mode="constant",
                        preserve_range=True,
                    ),
                    axis=-1,
                )
                mask = np.maximum(mask, mask_)

            self.y_train[n] = mask

    def predict_x(self, test=False):
        self.x_predicts = np.zeros(
            (len(self.test_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8
        )
        sizes_test = []
        print("Resizing test images")
        for n, id_ in tqdm(enumerate(self.test_ids), total=len(self.test_ids)):
            path = TEST_PATH + id_
            img = imread(path + "/images/" + id_ + ".png")[:, :, :IMG_CHANNELS]
            sizes_test.append([img.shape[0], img.shape[1]])
            img = resize(
                img, (IMG_HEIGHT, IMG_WIDTH), mode="constant", preserve_range=True
            )
            self.x_predicts[n] = img
        if test:
            return sizes_test
