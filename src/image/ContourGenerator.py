import math
import cv2
import numpy as np
from shapely.geometry import Polygon, MultiPolygon


class ContourGenerator:
    def __init__(self):
        pass

    def add_contours(self, target, mask_resized, colour=(0, 255, 0)):
        """
        Create contoured outline from mask image, and superimpose on target image
        :param target:
        :param mask_resized:
        :param colour:
        :return:
        """
        # Resize mask to match target, with range 0..255
        result = np.copy(target)
        levels = [0.2, 0.4, 0.6, 0.8]
        thicknesses = [1, 1, 1, 2]
        dict_all_contours = {}

        for i, level in enumerate(levels):
            contours = self.add_contours_to_image(result, mask_resized,
                                                  threshold=level, colour=colour, thickness=thicknesses[i])
            dict_all_contours[level] = contours

        # Fix alpha channel as necessary
        return result, dict_all_contours

    def add_contours_to_image(self, result, mask_resized, threshold, colour, thickness):
        contours = self.find_contours(mask_resized, threshold)

        # fcolour = self.get_contour_colour(colour, threshold, all_thresholds)
        fcolour = colour

        # Add to image
        cv2.drawContours(result, contours, -1, colour, thickness)
        return contours

    @staticmethod
    def find_contours(mask_resized, threshold, mode=cv2.RETR_EXTERNAL):
        # Apply thresholding for binary image, with hottest part as white
        binary_img = np.where(mask_resized > int(threshold * 255), 255, 0).astype(np.uint8)

        # Get contours
        contours_, _ = cv2.findContours(binary_img, mode, cv2.CHAIN_APPROX_SIMPLE)
        return contours_

    @staticmethod
    def get_contour_colour(colour, threshold, all_thresholds):
        # Adjust colour according to threshold represented
        fcolour = np.asarray(colour, dtype=np.float32)

        if len(all_thresholds) > 1:
            fcolour *= (threshold - all_thresholds[0]) / (all_thresholds[-1] - all_thresholds[0])

        if threshold == all_thresholds[-1]:
            fcolour = (255, 255, 255)

        fcolour = (int(fcolour[0]), int(fcolour[1]), int(fcolour[2]))  # for tuple compatible with cv2.drawContours
        return fcolour

    @staticmethod
    def generate_contour_stats(contours, patch_centre_xy):
        centroid_from_moments = lambda m: None if not m['m00'] else (int(m['m10'] / m['m00']), int(m['m01'] / m['m00']))

        contour_centroids = [centroid_from_moments(cv2.moments(c)) for c in contours]
        contour_areas = [cv2.contourArea(c) for c in contours]

        get_distance = lambda pt: math.sqrt((pt[0] - patch_centre_xy) ** 2 + (pt[1] - patch_centre_xy) ** 2)
        contour_centre_distances = [get_distance(cc) for cc in contour_centroids if cc is not None]

        return contour_areas, contour_centroids, contour_centre_distances

    @staticmethod
    def get_contour_union(contours):
        # Convert contour results to Shapely MultiPolygon
        # https://stackoverflow.com/questions/57965493/how-to-convert-numpy-arrays-obtained-from-cv2-findcontours-to-shapely-polygons, user Georgy
        cleaned_contours = [c for c in map(np.squeeze, contours) if len(c) > 2]  # remove redundant dimensions and stray points
        polygons = list(map(Polygon, cleaned_contours))  # converting to Polygons
        union = MultiPolygon(polygons)  # putting it all together in a MultiPolygon
        return union
