from xml.etree.ElementTree import parse

from classes.annotation.Box import ParentBox, Box


class ImageNetBoundingBoxLoader:

    def load_boxes(self, box_xml_path):
        root = parse(box_xml_path).getroot()
        bndboxes = root.findall("object/bndbox")
        boxes = [self.make_box(bndbox, i + 1) for i, bndbox in enumerate(bndboxes)]

        outer_box = self.make_outer_box(root, boxes)
        return outer_box

    def make_outer_box(self, root, boxes):
        size_node = root.find("size")
        h = self.findint(size_node, "height")
        w = self.findint(size_node, "width")
        outer_box = ParentBox(0, 0, 0, w, h, 0, boxes)
        return outer_box

    findint = lambda self, rootnode, nodename: int(rootnode.findtext(nodename))

    def make_box(self, bndbox, idx):
        xmin = self.findint(bndbox, "xmin")
        ymin = self.findint(bndbox, "ymin")
        xmax = self.findint(bndbox, "xmax")
        ymax = self.findint(bndbox, "ymax")

        box = Box(idx, xmin, ymin, xmax, ymax)
        return box
