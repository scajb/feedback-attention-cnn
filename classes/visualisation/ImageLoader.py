import os

import numpy as np
import torch
from skimage.io import imread
from skimage.transform import resize

from logging_support import log_info


class ImageLoader:
    def __init__(self):
        pass


    @staticmethod
    def load_image(input_path, device):
        log_info(f"Loading input image from {os.path.abspath(input_path)}")

        # Read from file as NumPy vector
        img = imread(input_path)
        orig_size = img.shape[0:2]
        torch_img = ImageLoader.generate_resized_rgb_image(device, img)
        return torch_img, orig_size

    @staticmethod
    def generate_resized_rgb_image(device, img, default_patch_size=(224, 224)):
        img = resize(img[:, :, 0:3], default_patch_size, preserve_range=True)  # discard alpha term from RGBA image from PNG
        img_vec = np.expand_dims(img.transpose((2, 0, 1)), 0)

        # Normalise
        img_vec /= 255.0
        mean = np.array([0.485, 0.456, 0.406]).reshape((1, 3, 1, 1))  # todo - get std, mean from CNN/trainer class
        std = np.array([0.229, 0.224, 0.225]).reshape((1, 3, 1, 1))
        norm_img = (img_vec - mean) / std

        torch_img = torch.from_numpy(norm_img).to(device, torch.float32)
        return torch_img

    @staticmethod
    def load_resized_torch_rgb_image(input_path, device, required_size):
        # Read image
        log_info(f"Loading torch RGB image from {os.path.abspath(input_path)}")
        img = imread(input_path)

        cropped_img = ImageLoader.square_crop(img)

        torch_img = ImageLoader.generate_resized_rgb_image(device, cropped_img, required_size)
        return torch_img, required_size

    @staticmethod
    def square_crop(img):
        # Crop to shorter dimension
        h = img.shape[0]
        w = img.shape[1]
        if h > w:
            h_margin = 0
            v_margin = int(0.5 * (h - w))
        else:
            h_margin = int(0.5 * (w - h))
            v_margin = 0

        cropped_img = img[v_margin:h - v_margin, h_margin:w - h_margin]
        return cropped_img

    @staticmethod
    def load_resized_numpy_float_image(input_path, required_size):
        log_info(f"Loading numpy float RGB image from {os.path.abspath(input_path)}")
        img = imread(input_path).astype('float64')[:, :, 0:3]  # RGBA->RGB
        cropped_img = ImageLoader.square_crop(img)
        return resize(cropped_img, required_size, preserve_range=True)


