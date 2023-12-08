import os
import sys
from pathlib import Path

import numpy as np
import skimage
from skimage.transform import resize
from skimage.io import imsave
import cv2 as cv

from classes.classifier.FeedbackAttentionLadderCNN import FeedbackAttentionLadderCNN
from classes.classifier.SaccadingFeedbackCNNContainer import ContourSaccadingFeedbackCNNContainer
from classes.filesystem.DirectorySupport import DirectorySupport
from classes.image.ContourGenerator import ContourGenerator
from classes.visualisation.ImageLoader import ImageLoader
from device import get_device
from logging_support import init_logging, log_info


def normalise_image(norm_img):
    norm_img = norm_img - norm_img.min()
    mx = norm_img.max()
    if mx > 0:
        norm_img /= mx
    return norm_img

def add_cross(img, coords, colour_, thickness=2, arm_len=10):
    x = int(img.shape[1] / 2 + coords[0])  # coords passed are relative to image centre, so offset them
    y = int(img.shape[0] / 2 + coords[1])
    img = cv.line(img, (x + arm_len, y), (x - arm_len, y), colour_, thickness=thickness)
    img = cv.line(img, (x, y + arm_len), (x, y - arm_len), colour_, thickness=thickness)
    return img

def save_resized_image(img_tile, size, output_path):
    resized_img = resize(img_tile, size, preserve_range=True)
    imsave(output_path, resized_img, check_contrast=False)
    log_info(f"Image {img_tile.shape} saved at {resized_img.shape} as: {os.path.abspath(output_path)}")

def execute_saccade_model():
    # Derive absolute file path and other parameters from shell args
    model_weights_path, image_path, log_path, output_dir_path = \
        [os.path.abspath(p) for p in sys.argv[1:5]]

    num_saccades = int(sys.argv[5])

    init_logging(log_path, sys.argv)

    # Load device: CUDA GPU if available, or CPU
    device = get_device(use_cpu=True)

    # Create FAL-CNN model and load pre-trained feedback weights from given file path
    fal_cnn_model = FeedbackAttentionLadderCNN.build_from_weights(device, model_weights_path)
    fal_cnn_input_size = (224, 224)

    # Create Saccade model with embedded FAL-CNN model
    saccade_model = ContourSaccadingFeedbackCNNContainer(None, num_saccades, device, fal_cnn_model)
    log_info("Created Saccade Model")

    # Load image file at 448x448px
    size_448px = (448, 448)
    img_448px = ImageLoader.load_resized_torch_rgb_image(image_path, device, size_448px)

    # Apply image to saccade model
    _, saccade_box_groups, feedback_groups, cropped_images, polygon_groups, centroids, class_outputs = \
        saccade_model(img_448px)

    # Plot sequence of cropped images with superimposed contours
    plot_saccade_images(centroids, class_outputs, cropped_images, fal_cnn_input_size, feedback_groups,
                        image_path, num_saccades, output_dir_path, saccade_box_groups, size_448px)


def plot_saccade_images(centroids, class_outputs, cropped_images, fal_cnn_input_size, feedback_groups,
                        image_path, num_saccades, output_dir_path, saccade_box_groups, size_448px):

    # Ensure output directory exists
    DirectorySupport.create_directory(output_dir_path)

    # Load image another image copy to superimpose with saccade outlines
    image_with_saccades = ImageLoader.load_resized_numpy_float_image(image_path, size_448px)

    contour_generator = ContourGenerator()
    box_colours = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255), (255, 0, 255), (255, 255, 0)]
    file_stem = Path(image_path).stem

    for i, saccade_image_batches in enumerate(cropped_images):
        # Convert cropped image from saccade sequence returned by model, to normalised UInt8 array to save as RGB image
        saccade_crop_image = saccade_image_batches[0].detach().cpu().numpy().transpose((1, 2, 0))
        saccade_crop_image = (normalise_image(saccade_crop_image) * 255).astype(np.uint8).copy()

        # Get mean feedback activations at FAL-CNN layer 28 (highest level feedback) and plot as contours on image
        layer_28_acts = feedback_groups[i][0][-1].detach().cpu().numpy()
        mean_fb_acts = np.mean(layer_28_acts[0], axis=0)
        resized_mean_fb = skimage.transform.resize(mean_fb_acts, fal_cnn_input_size, preserve_range=True)
        contoured_image, _ = contour_generator.add_contours(saccade_crop_image, normalise_image(resized_mean_fb) * 255)

        # Add cross icons at attention centroid and at the centre of the cropped image
        contoured_image = add_cross(contoured_image, centroids[i][0], (0, 255, 0))
        contoured_image = add_cross(contoured_image, (0, 0), (255, 0, 0))

        # Extract predicted class for this saccade and use in output filename
        predicted = class_outputs[i].detach().cpu().numpy()[0]
        saccade_crop_filename = f"{file_stem}-saccade-{i}-predicted-{predicted}.png"
        saccade_crop_path = os.path.join(output_dir_path, saccade_crop_filename)
        save_resized_image(contoured_image, fal_cnn_input_size, saccade_crop_path)

        # Draw saccade sampling boxes onto copy of 448x448px input image
        centres = []
        get_box_colour = lambda idx: box_colours[idx % len(box_colours)]

        for s, saccade_box_batches in enumerate(saccade_box_groups):
            box = saccade_box_batches[0]
            centres.append(box.get_int_centre())
            colour = get_box_colour(s)
            cv.rectangle(image_with_saccades, (box.left, box.top), (box.right, box.bottom), colour, 2)
            cv.putText(image_with_saccades, f"{s}",
                       (box.left - 30, box.top + 36 + int(188 * s / len(saccade_box_groups))),
                       cv.FONT_HERSHEY_SIMPLEX, 1.0, colour, 2)

        # Draw arrows showing saccade progression
        for c in range(len(centres) - 1, 0, -1):
            colour = get_box_colour(c)
            cv.arrowedLine(image_with_saccades, centres[c - 1], centres[c], colour, thickness=2, tipLength=0.25)

        # Write image file with superimposed boxes
        saccade_img_path = os.path.join(output_dir_path, f"{file_stem}-{num_saccades}-saccades.png")
        imsave(saccade_img_path, image_with_saccades, check_contrast=False)
        log_info(f"Saccade image {image_with_saccades.shape} saved as: {os.path.abspath(saccade_img_path)}")


if __name__ == "__main__":
    """
    ExecuteSaccadeModel.py

    Constructs and executes a Saccade model, with embedded FAL-CNN feedback attention model, and applies to specified 
    image file over specified number of iterations. Resizes and crops input image to 448x448px and samples 224x224px 
    region to feed to FAL-CNN. This region is initially taken from the centre of the larger image, then tracks to 
    follow centre of attention (CoA) derived from mean feedback activations at layer 28 of the FAL-CNN.   
    
    Outputs visualisation plots showing sampled regions with 80% attention contours and centroids, for initial region 
    and each subsequent saccade.

    Args:

    0) Path to this script
    1) Path to pre-trained weights for FAL-CNN model being used in saccade process 
    2) Path to RGB image to load and process
    3) Path for log file output 
    4) Output directory path for feedback visualisation plots
    5) Number of saccade iterations required

    """
    execute_saccade_model()