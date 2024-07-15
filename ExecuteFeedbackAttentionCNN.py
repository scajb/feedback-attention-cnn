import glob
import os
import sys

import cv2
import numpy as np
import skimage
import torch
from matplotlib import cm
from scipy import ndimage
from skimage.io import imsave

from src.annotation.ImageNetBoundingBoxLoader import ImageNetBoundingBoxLoader
from src.classifier.FeedbackAttentionLadderCNN import FeedbackAttentionLadderCNN
from src.filesystem.DirectorySupport import DirectorySupport
from src.image.ContourGenerator import ContourGenerator
from src.visualisation.ImageLoader import ImageLoader
from src.device import get_device
from src.logging_support import log_info, init_logging


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
    imsave(output_path, img.astype(np.uint8), check_contrast=False)
    log_info(f"Output image saved to: {os.path.abspath(output_path)}")


def resize_mask_to_target(target, mask, mask_range=1.0):
    zoom_ratio = int(target.shape[0] / mask.shape[0])
    mask_resized = ndimage.zoom(mask, zoom_ratio, order=0)
    return mask_resized / mask_range


def load_bounding_boxes(bounding_box_xml_dir, filename_stub, required_size):
    # Find XML annotation file e.g. n00007846_33974.xml corresponding to input filename e.g. ILSVRC2012_val_00033974
    image_file_id = int(filename_stub.split("_")[-1])  # convert to int to lose preceding zeros
    xml_file_wildcard = os.path.join(bounding_box_xml_dir, f"*_{image_file_id}.xml")
    matching_xml_files = glob.glob(xml_file_wildcard)

    if len(matching_xml_files) == 0:
        log_info(f"No bounding box found at {xml_file_wildcard} to match image {filename_stub}")
        return None

    # Load annotation - assume only one file matches unique id
    bb_file = matching_xml_files[0]
    log_info(f"Loaded bounding box annotation from {bb_file}")
    outer_box = ImageNetBoundingBoxLoader().load_boxes(bb_file)

    # Scale/crop bounding box coordinates according to image size going into model
    scaled_outer_box = outer_box.crop_scale_copy(required_size)

    # Return collection of Box objects for bounding box annotations
    return scaled_outer_box.child_boxes


def save_bounding_box_image(bboxes, output_dir_path, filename_stub, np_img):
    if bboxes is None:
        return

    colours = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255), (255, 0, 255), (255, 255, 0)]

    # Copy input image, to overwrite with bounding box outline
    image_with_bb = np_img.copy()

    # Add bounding box rectangles to new image
    for i, bb in enumerate(bboxes):
        cv2.rectangle(image_with_bb, (bb.left, bb.top), (bb.right, bb.bottom), colours[i % len(colours)], 1)

    # Save image
    filename = f"{filename_stub}-bounding-box.png"
    output_path = os.path.join(output_dir_path, filename)
    imsave(output_path, image_with_bb.astype(np.uint8), check_contrast=False)
    log_info(f"Output image saved to: {os.path.abspath(output_path)}")


def execute_feedback_attention():
    # Derive absolute file paths from shell args
    model_weights_path, image_path, log_path, output_dir_path = [os.path.abspath(p) for p in sys.argv[1:5]]
    bounding_box_xml_dir_path = "" if len(sys.argv) < 6 else sys.argv[5]

    init_logging(log_path, sys.argv)

    # Load device: CUDA GPU if available, or CPU
    device = get_device()  # (use_cpu=True) # use this option to force usage of CPU e.g. when GPU RAM insufficient.

    # Create FAL-CNN model and load pre-trained feedback weights from given file path
    model = FeedbackAttentionLadderCNN.build_from_weights(device, model_weights_path)

    # Load specified input image, as 224 x 224 pixel RGB torch tensor to fit model input
    required_size = (224, 224)
    torch_img = ImageLoader.load_resized_torch_rgb_image(image_path, device, required_size)

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

    # Load and plot ground truth bounding box(es), if available for current image
    bboxes = load_bounding_boxes(bounding_box_xml_dir_path, filename_stub, required_size)
    save_bounding_box_image(bboxes, output_dir_path, filename_stub, np_img)

    # For each feedback iteration reported by model
    for iteration_num, feedback_iteration_results in enumerate(feedback_iterations):
        feedbacks = [fb.detach().cpu().numpy() for fb in feedback_iteration_results]

        # For each feedback activation (to each CNN layer where feedback applied)
        for l, batch in enumerate(feedbacks):
            feedback_activations = batch[0]  # only one image in batch here
            feedback_layers = [0, 5, 10, 19, 28]  # feedback layers used in 'hybrid recurrent' feedback model
            feedback_layer_num = feedback_layers[l]

            # Normalise, scale and combine feedback activations into a mean 'heatmap' array for current feedback layer
            heatmap_img = create_feedback_heatmap(feedback_activations, feedback_layer_num, required_size)

            # Plot and save attention heatmaps (mean of feedback activations across all channels)
            # save_image(output_dir_path, filename_stub, feedback_layer_num, iteration_num, "heatmap", heatmap_img)

            # Plot and save contour image
            resized_heatmap_image = resize_mask_to_target(np_img, normalise_image(heatmap_img) * 255)
            contoured_image, all_contours = ContourGenerator().add_contours(np_img, resized_heatmap_image)
            save_image(output_dir_path, filename_stub, feedback_layer_num, iteration_num,
                       "contours", contoured_image, "jpeg")


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
    1) Path to pre-trained weights for FAL-CNN model under test
    2) Path to RGB image to load and process
    3) Path for log file output 
    4) Output directory path for feedback visualisation plots
    5) Optional: Directory path for ImageNet bounding box annotation XML
    
    """
    execute_feedback_attention()
