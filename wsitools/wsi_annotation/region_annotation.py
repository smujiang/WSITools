import numpy as np
import openslide
from xml.dom import minidom
from shapely.geometry.polygon import Polygon
from shapely.geometry import box
from shapely.geometry import Point
import matplotlib.pyplot as plt

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
            print("TODO: create polyline region")
        elif shape == "Ellipse":
            print("TODO: create ellipse region")
        elif shape == "Rectangle":
            # self.geo_region = box(points)
            self.geo_region = Polygon(points)
        else:
            print("Not a region")


class AnnotationRegions:
    def __init__(self, xml_fn):
        xml = minidom.parse(xml_fn)
        regions_dom = xml.getElementsByTagName("Region")
        self.Regions = []
        points = np.empty([0, 2], dtype=np.float)
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
            points = np.vstack([points, coords])
            self.Regions.append(Region(points, region_geo_shape, region_Id, class_label_Id, class_label_text))

    #TODO: deal with multiple labels
    def get_patch_label(self, patch_loc):  # patch location should be top left
        point = Point(patch_loc)
        for idx, region in enumerate(self.Regions):
            if region.shape == "Polygon":
                if point.within(region.geo_region):
                    print("Region ID: %s, Label ID: %s, Label text: %s, Shape: %s" % (region.region_id, region.label_id, region.label_text, region.shape))
                    return region.label_id, region.label_text

    # TODO: deal with multiple labels
    def create_patch_annotation_mask(self, patch_loc, patch_size): # patch location should be top left
        mask_array = np.zeros([patch_size, patch_size], dtype=np.uint8)
        for w in range(patch_size):
            for h in range(patch_size):
                point = Point([patch_loc[0]+w, patch_loc[1]+h])
                for idx, region in enumerate(self.Regions):
                    if region.shape == "Polygon":
                        if point.within(region.geo_region):
                            mask_array[h, w] += int(region.label_id)
                            #print("Region ID: %s, Label ID: %s, Label text: %s, Shape: %s" % (region.region_id, region.label_id, region.label_text, region.shape))
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
    anno_regions = AnnotationRegions(xml_fn)
    # point = [105910.148438, 54728.425781]
    # anno_regions.get_patch_label(point)
    micron_loc = [8451.17, 6240.97]
    pix_loc = anno_regions.convert_micron_coord_2_pixel_coord(micron_loc)
    anno_regions.validate_annotation(wsi_fn, pix_loc, patch_size=512)









