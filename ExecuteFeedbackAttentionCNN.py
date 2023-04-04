import os
import sys

import torch

from classes.visualisation.ImageLoader import ImageLoader
from device import get_device
from logging_support import log_info, init_logging


def execute_feedback_attention():
    # Derive absolute file paths from shell args
    model_path, image_path, log_path = [os.path.abspath(p) for p in sys.argv[1:4]]
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

    # Apply model to image
    class_out, feedback_iterations = model(torch_img)

    # Report class prediction
    _, cls = torch.max(class_out.data, 1)
    predicted_class_idx = cls.detach().cpu().numpy()[0]
    log_info(f"Model predicted class {predicted_class_idx} for image {image_path}")

    # Plot and save contour image

    # Plot and save attention heatmap (mean of feedback activation)


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
