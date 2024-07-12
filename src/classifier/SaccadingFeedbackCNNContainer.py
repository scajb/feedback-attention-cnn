import random

import numpy as np
import torch
from scipy.ndimage import measurements
from skimage.transform import resize
from torch import nn
from torch.autograd import Variable

from src.annotation.Box import Box
from src.image.ContourGenerator import ContourGenerator


class SaccadingFeedbackCNNContainer(nn.Module):
    def __init__(self, fb_model_path, num_saccades, device, fb_model=None):
        """
        Container module for feedback attention modules. Manages saccade-like process of selecting
        input patches from larger images, applying feedback CNN model, then re-sampling
        inputs centred around centroid of feedback activations.
        """
        super(SaccadingFeedbackCNNContainer, self).__init__()
        self.num_saccades = num_saccades
        self.device = device

        # Use existing model - must be correctly pre-initialised
        if fb_model is not None:
            self.feedback_model = fb_model

        # Load feedback model from file
        elif fb_model_path:
            self.feedback_model = torch.load(fb_model_path, map_location=device)
            self.feedback_model.device = device  # force device to match
            self.feedback_model.eval()

    def sample_patches(self, out, sampling_offsets, output_patch_hw):
        """
        Samples square patches of specified height and width, from each image in batch, with
        a distinct x,y offset for each item
        :param out:
        :param sampling_offsets: list of x,y offsets per image in batch
        :param output_patch_hw:
        :return: PyTorch tensor of sampled patch images, one per item in batch
        """
        sampled_patches = torch.zeros((out.shape[0], out.shape[1], output_patch_hw, output_patch_hw))
        input_size = out.shape[2]
        boxes = []

        for b, batched_image in enumerate(out):
            offset_xy = sampling_offsets[b]
            x1 = offset_xy[0] + int(input_size / 2) - int(output_patch_hw / 2)
            x1 = min(max(x1, 0), input_size - output_patch_hw)
            x2 = x1 + output_patch_hw
            y1 = offset_xy[1] + int(input_size / 2) - int(output_patch_hw / 2)
            y1 = min(max(y1, 0), input_size - output_patch_hw)
            y2 = y1 + output_patch_hw
            boxes.append(Box(b, x1, y1, x2, y2))

            for r, _ in enumerate(batched_image):
                sampled_patches[b, r] = out[b, r, y1:y2, x1:x2]

        return Variable(sampled_patches.to(self.device)), boxes

    def get_feedback_centroids(self, feedbacks, output_patch_hw, previous_offsets):
        """
        Returns list of feedback centroids, as centres of mass of each feedback activation for batch
        :param feedbacks: feedback activations for multiple layers, from feedback model's forward() call,
        one set per image in batch
        :param output_patch_hw: height and width of output patch (224px here)
        :param previous_offsets: previous resultsombine with
        :return:
        """
        # get last layer (layer 28) for first (and only) feedback iteration, batched for all images
        layer_28_fb_activations = feedbacks[0][-1].tolist()

        # First dimension in this tensor is image within batch. Need centroid for each
        centroids_and_polygons = [self.get_feedback_centroid(im, output_patch_hw) for im in layer_28_fb_activations]
        centroids = [cp[0] for cp in centroids_and_polygons]
        polygons = [cp[1] for cp in centroids_and_polygons]

        # Add centroid coordinates to previous offsets; these determine the absolute position of each point
        polygon_centroids = centroids.copy()
        self.add_previous_offfsets(centroids, previous_offsets)
        return centroids, polygons, polygon_centroids

    @staticmethod
    def add_previous_offfsets(centroids, previous_offsets):
        for i, centroid in enumerate(centroids):
            add_point = lambda pt1, pt2: (pt1[0] + pt2[0], pt1[1] + pt2[1])
            centroids[i] = add_point(centroid, previous_offsets[i])

    def get_feedback_centroid(self, fb_activations, output_patch_hw):
        mean_activation_map = np.mean(fb_activations, axis=0)

        # Normalise
        norm_act_map = mean_activation_map - mean_activation_map.min()
        norm_act_map /= norm_act_map.max()

        # Resize to match model input (224x224px)
        norm_act_map = resize(norm_act_map, (output_patch_hw, output_patch_hw), preserve_range=True)

        centroid_x, centroid_y, multipolygon = self.get_image_centroid(norm_act_map)

        scale_to_output = lambda centroid_pos, old_size, new_size: int(centroid_pos * new_size / old_size) - \
                                                                   int(output_patch_hw / 2)  # offset so 0,0 is middle

        centroid_x_scaled = scale_to_output(centroid_x, norm_act_map.shape[1], output_patch_hw)
        centroid_y_scaled = scale_to_output(centroid_y, norm_act_map.shape[0], output_patch_hw)
        return [centroid_x_scaled, centroid_y_scaled], multipolygon

    def get_image_centroid(self, norm_act_map):
        centroid_y, centroid_x = measurements.center_of_mass(norm_act_map)  # note order of dimensions: h, w
        return centroid_x, centroid_y, None,

    def forward(self, out):
        out.requires_grad = True

        # Validate input h, w in tensor shaped as (batch, rgb, h, w)
        if out.shape[2] != 448 or out.shape[3] != 448:
            raise ValueError("Expecting 448x448px images in input batch")

        # Define a sampling location per image in input batch, initially at the centre
        get_rand_int = lambda: random.randint(-22, 22)  # approx +/- 10% of patch width
        initial_offset = (0, 0)  # (get_rand_int(), get_rand_int())
        sampling_offsets = [initial_offset for b in range(out.shape[0])]

        # Sample central 224px patch from expected 448x448px images
        output_patch_hw = 224
        sampled_patches, boxes = self.sample_patches(out, sampling_offsets, output_patch_hw)
        saccade_box_groups = [boxes]
        cropped_images = [sampled_patches]

        # Apply feedback model to sampled patches
        final_out, feedbacks = self.feedback_model(sampled_patches)
        feedback_groups = [feedbacks]
        output_class_groups = [torch.max(final_out, 1)[1]]
        polygon_groups = []
        centroid_groups = []

        for s in range(self.num_saccades):
            # Calculate centres of mass of feedback activations
            sampling_offsets, polygons, centroids = self.get_feedback_centroids(feedbacks, output_patch_hw,
                                                                                sampling_offsets)
            polygon_groups.append(polygons)
            centroid_groups.append(centroids)

            # Sample new patches, offset according to new centroids
            sampled_patches, boxes = self.sample_patches(out, sampling_offsets, output_patch_hw)
            saccade_box_groups.append(boxes)
            cropped_images.append(sampled_patches)

            # Apply feedback model again, to recentred patches
            final_out, feedbacks = self.feedback_model(sampled_patches)
            feedback_groups.append(feedbacks)
            output_class_groups.append(torch.max(final_out, 1)[1])

        # Just to get last contour
        sampling_offsets, polygons, centroids = self.get_feedback_centroids(feedbacks, output_patch_hw,
                                                                            sampling_offsets)
        polygon_groups.append(polygons)
        centroid_groups.append(centroids)

        return final_out, \
               saccade_box_groups, feedback_groups, cropped_images, polygon_groups, centroid_groups, output_class_groups


class ContourSaccadingFeedbackCNNContainer(SaccadingFeedbackCNNContainer):
    def __init__(self, fb_model_path, num_saccades, device, fb_model=None):
        """
        Subclass of SaccadingFeedbackCNNContainer which uses largest 80% contour to derive next saccade point
        """
        super(ContourSaccadingFeedbackCNNContainer, self).__init__(fb_model_path, num_saccades, device, fb_model)

    def get_image_centroid(self, norm_act_map):

        contours = ContourGenerator.find_contours(norm_act_map * 255, threshold=0.8)
        contour_multipolygon = ContourGenerator.get_contour_union(contours)
        largest_contour = None if contour_multipolygon.is_empty else max(contour_multipolygon.geoms, key=lambda a: a.area)
        if largest_contour is not None:
            return largest_contour.centroid.x, largest_contour.centroid.y, largest_contour
        else:
            return int(norm_act_map.shape[0] / 2), int(norm_act_map.shape[1] / 2), largest_contour
            # default to centre if no contours found


class SaccadingModalResultFeedbackCNNContainerMixin:
    def forward(self, out):
        final_out, saccade_box_groups, feedback_groups, cropped_images, polygon_groups, \
        centroid_groups, output_class_groups = super().forward(out)

        # Want final out class to be modal result from all saccades
        modal_class_out = torch.zeros_like(final_out)
        for b, _ in enumerate(final_out):
            for og in output_class_groups:
                saccade_class = og[b]
                modal_class_out[b, saccade_class] += 1.

        return modal_class_out, saccade_box_groups, feedback_groups, cropped_images, polygon_groups, centroid_groups, \
               output_class_groups


class ContourSaccadingModalResultFeedbackCNNContainer(ContourSaccadingFeedbackCNNContainer,
                                                      SaccadingModalResultFeedbackCNNContainerMixin):
    pass


class ComSaccadingModalResultFeedbackCNNContainer(SaccadingFeedbackCNNContainer,
                                                  SaccadingModalResultFeedbackCNNContainerMixin):
    pass


class RandomSaccadingFeedbackCNNContainer(SaccadingFeedbackCNNContainer):
    def __init__(self, fb_model_path, num_saccades, device, rand_range_xy: int, fb_model=None):
        """
        Subclass of SaccadingFeedbackCNNContainer which makes a random saccade each time, to compare the effect
        of attention-led and random patch sampling offsets
        """
        super(RandomSaccadingFeedbackCNNContainer, self).__init__(fb_model_path, num_saccades, device, fb_model)
        self.rand_range_xy = rand_range_xy

    def get_image_centroid(self, norm_act_map):
        offset_h = random.randint(-self.rand_range_xy, self.rand_range_xy)
        offset_w = random.randint(-self.rand_range_xy, self.rand_range_xy)

        return int(norm_act_map.shape[0] / 2) + offset_h, \
               int(norm_act_map.shape[1] / 2) + offset_w, \
               None
