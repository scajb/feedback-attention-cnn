from shapely.ops import unary_union
from logging_support import log_info
from shapely.geometry import Polygon


class Box:
    """
    Box class. Represents single image patch within Whole Slide Image, as defined in QUASAR Box XML file,
    of specified size and location.

    NEEDS TO STAY IN CLASSES ROOT DIR TO SUPPORT PREVIOUSLY SERIALIZED BOXES
    """

    def __init__(self, index, left, top, right, bottom, classification=None, size=100,
                 parent_left=0, parent_top=0, parent_size=(0, 0), plot_colour=None):
        self.left = left
        self.right = right
        self.top = top
        self.bottom = bottom
        self.index = index
        self.classification = classification
        self.size = size
        self.parent_left = parent_left
        self.parent_top = parent_top
        self.parent_size = parent_size
        self.plot_colour = plot_colour

    @classmethod
    def from_vertices(cls, index, vertices_x, vertices_y, classification, size):
        x = min(vertices_x)
        y = min(vertices_y)
        return Box.from_centre_and_size(index, x, y, classification, size)

    @classmethod
    def from_bounding_box_vertices(cls, vertices_x, vertices_y):
        return Box(0,
                   int(min(vertices_x)),
                   int(min(vertices_y)),
                   int(max(vertices_x)),
                   int(max(vertices_y)))

    @classmethod
    def from_centre_and_size(cls, index, x, y, classification, size):
        half_size = int(size) / 2
        return cls(
            index=index,
            left=x - half_size,
            right=x + half_size,
            top=y - half_size,
            bottom=y + half_size,
            classification=classification,
            size=size)

    @classmethod
    def scaled_copy(cls, box, scale):
        return cls(
            index=box.index,
            left=box.left * scale[0],
            right=box.right * scale[0],
            top=box.top * scale[1],
            bottom=box.bottom * scale[1],
            plot_colour=box.plot_colour,
            classification=box.classification)

    @classmethod
    def offset_copy(cls, box, xy_offset):
        return cls(
            index=box.index,
            left=box.left + xy_offset[0],
            right=box.right + xy_offset[0],
            top=box.top + xy_offset[1],
            bottom=box.bottom + xy_offset[1],
            plot_colour=box.plot_colour,
            classification=box.classification,
            size=box.size)

    @classmethod
    def square_box_around_rect(cls, bounds):
        minx, miny, maxx, maxy = bounds
        w = abs(maxx - minx)
        h = abs(maxy - miny)
        centroid = [(maxx + minx) / 2, (maxy + miny) / 2]
        largest_dimension = w if w > h else h
        return Box.from_centre_and_size(0, centroid[0], centroid[1], "0", largest_dimension)

    def get_area(self):
        return self.get_width() * self.get_height()

    def get_vertices(self):
        return [[self.left, self.top],
                [self.right, self.top],
                [self.right, self.bottom],
                [self.left, self.bottom]]

    def get_centre(self):
        return [self.left + (self.right - self.left) / 2,
                self.top + (self.bottom - self.top) / 2]

    def get_width(self):
        return self.right - self.left

    def get_height(self):
        return self.bottom - self.top

    def to_shapely_polygon(self, size=0):

        if size == 0:
            vertices = self.get_vertices()

        else:
            cx, cy = self.get_centre()
            left = cx - size / 2
            right = cx + size / 2
            top = cy - size / 2
            bottom = cy + size / 2
            vertices = [[left, top],
                        [right, top],
                        [right, bottom],
                        [left, bottom]]

        return Polygon(vertices)

    def __str__(self):
        return "Box index={} left={} top={} right={} bottom={} classification={} size={}".format(
            self.index, self.left, self.top, self.right, self.bottom, self.classification, self.size)



class ParentBox(Box):

    def __init__(self, index, left, top, right, bottom, size, child_boxes):
        super().__init__(index, left, top, right, bottom, size=size)
        self.child_boxes = child_boxes

    @staticmethod
    def get_patches_and_bounds(parent_boxes):
        cell_patches = []
        max_x, max_y, max_count = 0, 0, 0
        for box in parent_boxes:
            cell_patches.extend(box.child_boxes)
            max_x = max(max_x, box.right)
            max_y = max(max_y, box.bottom)
            max_count = max(max_count, len(box.child_boxes))

        return (max_x, max_y), cell_patches, max_count

    def crop_scale_copy(self, crop_size):
        """
        Returns scaled, centre-cropped copy of this parent box and its child boxes
        :param crop_size: x, y size e.g. 224, 224px of required output
        :return:
        """
        this_size = (self.right - self.left, self.bottom - self.top)
        aspect = lambda size: 0 if size[1] == 0 else abs(size[0]/size[1])

        if aspect(crop_size) < aspect(this_size):
            # Scale according to relative height, then crop right and left
            scale_factor = crop_size[1] / this_size[1]
            h_crop = 0.5 * (this_size[0] - this_size[1])  # for square at current height
            v_crop = 0
        else:
            # Scale according to relative width, then crop top and bottom
            scale_factor = crop_size[0] / this_size[0]
            h_crop = 0
            v_crop = 0.5 * (this_size[1] - this_size[0])  # for square at current width

        def make_scaled_cropped_box(box):
            left = max(0, int((box.left - h_crop) * scale_factor))
            right = min(crop_size[0], int((box.right - h_crop) * scale_factor))
            top = max(0, int((box.top - v_crop) * scale_factor))
            bottom = min(crop_size[1], int((box.bottom - v_crop) * scale_factor))
            return Box(box.index, left, top, right, bottom, box.size)

        scaled_boxes = [make_scaled_cropped_box(bx) for bx in self.child_boxes]
        return ParentBox(self.index, 0, 0, crop_size[0], crop_size[1], crop_size[0], scaled_boxes)

    def union_child_boxes(self):
        """
        Returns Shapely Polygon for union of child box regions
        :return:
        """
        child_polygons = [b.to_shapely_polygon() for b in self.child_boxes]
        union = unary_union(child_polygons)
        return union

