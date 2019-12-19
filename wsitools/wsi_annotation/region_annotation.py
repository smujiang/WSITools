import numpy as np
import openslide
from xml.dom import minidom
from shapely.geometry.polygon import Polygon
from shapely.geometry import box
from shapely.geometry import Point
import matplotlib.pyplot as plt
import logging
from wsitools.file_management.class_label_csv_manager import ClassLabelCSVManager


class Region:
    def __init__(self, points, shape, region_id, label_id, label_text):
        self.shape = shape
        self.region_id = region_id
        self.label_id = label_id
        self.label_text = label_text
        if shape == "Polygon":
            self.geo_region = Polygon(points)
            # coords = [p.coords[:][0] for p in points]
            # poly = Polygon(points)
        elif shape == "Area":
            self.geo_region = Polygon(points)
        elif shape == "Polyline":
            logging.info("TODO: create polyline region")
        elif shape == "Ellipse":
            logging.info("TODO: create ellipse region")
        elif shape == "Rectangle":
            X = points[:, 0]
            Y = points[:, 1]
            self.geo_region = box(minx=min(X), miny=min(Y), maxx=max(X), maxy=max(Y))
        else:
            logging.info("Not a region")


class AnnotationRegions:
    def __init__(self, xml_fn, class_label_id_csv):
        xml = minidom.parse(xml_fn)
        self.class_label_id = ClassLabelCSVManager(class_label_id_csv)
        regions_dom = xml.getElementsByTagName("Region")
        self.Regions = []
        for reg_dom in regions_dom:
            vertices = reg_dom.getElementsByTagName("Vertex")
            region_Id = reg_dom.getAttribute('Id')
            class_label_text = reg_dom.getAttribute('Text')
            class_label_Id = reg_dom.getAttribute('Type')
            region_geo_shape = reg_dom.getAttribute('GeoShape')
            coords = np.zeros((len(vertices), 2))
            for i, vertex in enumerate(vertices):
                coords[i][0] = vertex.attributes['X'].value
                coords[i][1] = vertex.attributes['Y'].value
            self.Regions.append(Region(coords, region_geo_shape, region_Id, class_label_Id, class_label_text))

    def get_pixel_label(self, pixel_loc):  # patch location should be top left
        point = Point(pixel_loc)
        label_id = []
        label_text = []
        for idx, region in enumerate(self.Regions):
            # Currently, we only support these three geometric types
            if region.shape == "Rectangle" or region.shape == "Polygon" or region.shape == "Area":
                if point.within(region.geo_region):
                    # print("Region ID: %s, Label ID: %s, Label text: %s, Shape: %s" % (region.region_id, region.label_id, region.label_text, region.shape))
                    label_id.append(region.label_id)
                    label_text.append(region.label_text)
        if len(label_text) > 1:  # more than one label, choose the highest priority
            label_pri = []
            for lt in label_text:
                label_pri.append(self.class_label_id.get_priority(lt))
            l_idx = label_pri.index(max(label_pri))  # Be aware, there might be equal priority.
            if type(l_idx) == list:
                # print("Pixel warning, Equal priority, return the first one")
                return label_id[l_idx[0]], label_text[l_idx[0]]  # equal priority, choose the first one.
            else:
                return label_id[l_idx],  label_text[l_idx]
        elif len(label_text) == 0:
            return -1, "null"
        else:
            # print("get a labeled patch %s" % label_text[0])
            return label_id[0], label_text[0]

    # get a mask from annotation
    def create_patch_annotation_mask(self, patch_loc, patch_size):  # patch location should be top left
        mask_array = np.zeros([patch_size, patch_size], dtype=np.uint8)
        for w in range(patch_size):
            for h in range(patch_size):
                mask_array[h, w] = self.get_pixel_label([patch_loc[0]+w, patch_loc[1]+h])
        return mask_array

    #  get a mask from annotation for debugging
    def create_patch_annotation_mask_debug(self, patch_loc, patch_size):  # patch location should be top left
        mask_array = np.zeros([patch_size, patch_size], dtype=np.uint8)
        for w in range(patch_size):
            for h in range(patch_size):
                point = Point([patch_loc[0] + w, patch_loc[1] + h])
                for idx, region in enumerate(self.Regions):
                    if region.shape == "Rectangle" or region.shape == "Polygon" or region.shape == "Area":
                        if point.within(region.geo_region):
                            mask_array[h, w] += int(region.label_id)
        return mask_array

    def validate_annotation(self, wsi_fn, patch_loc, level=0, patch_size=256):
        wsi_obj = openslide.open_slide(wsi_fn)
        patch = wsi_obj.read_region(patch_loc, level, [patch_size, patch_size])
        ann_mask = self.create_patch_annotation_mask(patch_loc, patch_size)
        # ann_mask = np.zeros([patch_size, patch_size])
        fig = plt.figure()
        ax1 = fig.add_subplot(121)
        ax1.imshow(patch)
        ax2 = fig.add_subplot(122)
        ax2.imshow(ann_mask, cmap='jet')
        plt.show()

    @staticmethod
    def convert_micron_coord_2_pixel_coord(micron_coord, pixel_size=0.25):  # pixel size 0.25 um
        return (np.array(micron_coord)/pixel_size).astype(np.int32)


# example
if __name__ == "__main__":
    wsi_fn = "/projects/shart/digital_pathology/data/PenMarking/WSIs/MELF/e39a8d60a56844d695e9579bce8f0335.tiff"
    xml_fn = "/projects/shart/digital_pathology/data/PenMarking/annotations/temp/e39a8d60a56844d695e9579bce8f0335.xml"
    class_label_id_csv = "/projects/shart/digital_pathology/data/PenMarking/annotations/temp/label_id.csv"

    anno_regions = AnnotationRegions(xml_fn, class_label_id_csv)
    # point = [105910.148438, 54728.425781]
    # anno_regions.get_patch_label(point)
    micron_loc = [8451.17, 6240.97]
    pix_loc = anno_regions.convert_micron_coord_2_pixel_coord(micron_loc)
    anno_regions.get_pixel_label(pix_loc)   # pixel location is patch's top left
    anno_regions.validate_annotation(wsi_fn, pix_loc, patch_size=800)









