import os
import sys

import numpy as np
import skimage
import torch
from matplotlib import cm
from scipy import ndimage
from skimage.io import imsave

from classes.filesystem.DirectorySupport import DirectorySupport
from classes.image.ContourGenerator import ContourGenerator
from classes.visualisation.ImageLoader import ImageLoader
from device import get_device
from logging_support import log_info, init_logging


def normalise_image(norm_img):
    norm_img = norm_img - norm_img.min()
    mx = norm_img.max()
    if mx > 0:
        norm_img /= mx
    return norm_img

def apply_heatmap(combined_heatmap, map_name="jet"):
    cmap = cm.get_cmap(map_name, 256)
    heatmap_cmap = cmap(combined_heatmap)  # as RGBA
    heatmap_cmap = heatmap_cmap[:, :, 0: 3]  # RGB
    return heatmap_cmap

def normalise_feedback_activations(feedback_images, layer_num):
    if layer_num == 0:  # UNet output as layer 0 feedback, already an RGB image
        norm_img = normalise_image(feedback_images)
        norm_img = np.transpose(norm_img, (1, 2, 0))  # RGB->last dim
    else:
        # Multi-channel feedback between inner layers, shown as mean heatmap
        mean_img = np.mean(feedback_images, axis=0)
        heatmap_img = apply_heatmap(mean_img)
        norm_img = normalise_image(heatmap_img)
    return norm_img

def save_image(output_dir, filename_stub, layer_num, iteration_num, suffix, img, extn="jpeg"):
    # Write file to output dir
    filename = f"{filename_stub}-layer-{layer_num}-iteration-{iteration_num}-{suffix}.{extn}"
    output_path = os.path.join(output_dir, filename)
    imsave(output_path, img, check_contrast=False)
    log_info(f"Output image saved to: {os.path.abspath(output_path)}")

def resize_mask_to_target(target, mask, mask_range=1.0):
    zoom_ratio = int(target.shape[0] / mask.shape[0])
    mask_resized = ndimage.zoom(mask, zoom_ratio, order=0)
    return mask_resized / mask_range


def execute_feedback_attention():
    # Derive absolute file paths from shell args
    model_path, image_path, log_path, output_dir_path = [os.path.abspath(p) for p in sys.argv[1:5]]
    init_logging(log_path, sys.argv)

    # Load device: CUDA GPU if available, or CPU
    device = get_device(use_cpu=True)

    # Load pre-trained feedback attention CNN model from given file path
    log_info(f"Loading model from {model_path}")
    model = torch.load(model_path, map_location=device)
    model.device = device

    # Load specified input image, as 224 x 224 pixel RGB torch tensor to fit model input
    required_size = (224, 224)
    torch_img, _ = ImageLoader.load_resized_torch_rgb_image(image_path, device, required_size)

    # Apply model to image.
    # Model returns predicted class tensor, and nested collection of feedback activations per feedback iteration,
    # batch item, feedback layer, model channel, height and width dimensions
    class_out, feedback_iterations = model(torch_img)

    # Report class prediction
    _, cls = torch.max(class_out.data, 1)
    predicted_class_idx = cls.detach().cpu().numpy()[0]
    log_info(f"Model predicted class {predicted_class_idx} for image {image_path}")

    # Ensure output directory exists
    DirectorySupport.create_directory(output_dir_path)

    # Derive a file prefix to use for outputs
    filename_stub = os.path.splitext(os.path.basename(image_path))[0]

    # Load input image again, as a numpy array, to use in visualisations
    np_img = ImageLoader.load_resized_numpy_float_image(image_path, required_size)

    # For each feedback iteration reported by model
    for iteration_num, feedback_iteration_results in enumerate(feedback_iterations):
        feedbacks = [fb.detach().cpu().numpy() for fb in feedback_iteration_results]

        # For each feedback activation (to each CNN layer where feedback applied)
        for layer_idx, feedback_batch in enumerate(feedbacks):
            feedback_activations = feedback_batch[0]
            feedback_layers = [0, 5, 10, 19, 28]
            feedback_layer_num = feedback_layers[layer_idx]

            # Normalise, scale and combine feedback activations into a mean 'heatmap' array for current feedback layer
            heatmap_img = create_feedback_heatmap(feedback_activations, feedback_layer_num, required_size)

            # Plot and save attention heatmaps (mean of feedback activations across all channels)
            save_image(output_dir_path, filename_stub, feedback_layer_num, iteration_num, "heatmap", heatmap_img)

            # Plot and save contour image
            resized_heatmap_image = resize_mask_to_target(np_img, normalise_image(heatmap_img) * 255)
            contoured_image, all_contours = ContourGenerator().add_contours(np_img, resized_heatmap_image)
            save_image(output_dir_path, filename_stub, feedback_layer_num, iteration_num, "contours", contoured_image)


def create_feedback_heatmap(feedback_activations, feedback_layer_num, required_size):
    norm_fb_acts = normalise_feedback_activations(feedback_activations, feedback_layer_num)
    scaled_fb_acts = skimage.transform.resize(norm_fb_acts, required_size, preserve_range=True)
    heatmap_img = np.mean(scaled_fb_acts, axis=2)
    return heatmap_img


if __name__ == "__main__":
    """
    ExecuteFeedbackAttentionCNN.py
    
    Loads and executes a feedback attention CNN model, obtaining image class predictions and feedback activation
    tensors. 
    
    Plots various visualisations of the feedback activations, for comparison with ground truth bounding boxes and 
    other annotations. 
    
    Args:
    
    0) Path to this script
    1) Path to pre-trained CNN model under test
    2) Path to RGB image to load and process
    3) Path for log file output 
    
    """
    execute_feedback_attention()
