import os, math
import numpy as np
import openslide
from PIL import Image
import matplotlib.pyplot as plt
from xml.dom import minidom


class OffsetAnnotation:
    @staticmethod
    # attr_filter, please refer to: https://docs.python.org/2/library/xml.etree.elementtree.html
    def load_QuPath_points_from_xml(xml_fn, geo_shape='Points', label_text='offset'):
        xml = minidom.parse(xml_fn)
        regions = xml.getElementsByTagName("Region")
        points = np.empty([0, 2], dtype=np.float)
        for region in regions:
            vertices = region.getElementsByTagName("Vertex")
            label_text = region.getAttribute('Text')
            region_geo_shape = region.getAttribute('GeoShape')
            if region_geo_shape == geo_shape and label_text == label_text:
                # Store x, y coordinates into a 2D array in format [x1, y1], [x2, y2], ...
                coords = np.zeros((len(vertices), 2))
                for i, vertex in enumerate(vertices):
                    coords[i][0] = vertex.attributes['X'].value
                    coords[i][1] = vertex.attributes['Y'].value
                points = np.vstack([points, coords])
        return points

    # ------------------Way to get offset or affine matrix--------------------------- #
    def get_xml_offset_barycentric(self, fixed_xml, float_xml):
        template_anno_points = self.load_QuPath_points_from_xml(fixed_xml, geo_shape='Points', label_text='offset')
        test_anno_points = self.load_QuPath_points_from_xml(float_xml, geo_shape='Points', label_text='offset')
        offsets = np.mean(template_anno_points, axis=1) - np.mean(test_anno_points, axis=1)
        return offsets

    @staticmethod
    def get_affine_matrix(fixed_points, float_points):
        l = len(fixed_points)
        B = np.vstack([np.transpose(fixed_points), np.ones(l)])
        D = 1.0 / np.linalg.det(B)
        entry = lambda r, d: np.linalg.det(np.delete(np.vstack([r, B]), (d + 1), axis=0))
        M = [[(-1) ** i * D * entry(R, i) for i in range(l)] for R in np.transpose(float_points)]
        A, t = np.hsplit(np.array(M), [l - 1])
        t = np.transpose(t)[0]
        return A, t

    @staticmethod
    def validate_offset(fixed_wsi_fn, float_wsi_fn, offset, scale=100):
        template_wsi = openslide.open_slide(fixed_wsi_fn)
        WSI_Width, WSI_Height = template_wsi.dimensions
        thumb_size_x = int(WSI_Width / scale)
        thumb_size_y = int(WSI_Height / scale)
        template_thumbnail = template_wsi.get_thumbnail([thumb_size_x, thumb_size_y])

        test_wsi = openslide.open_slide(float_wsi_fn)
        WSI_Width, WSI_Height = test_wsi.dimensions
        thumb_size_xx = int(WSI_Width / scale)
        thumb_size_yy = int(WSI_Height / scale)
        test_thumbnail = test_wsi.get_thumbnail([thumb_size_xx, thumb_size_yy])

        thumb_offset = [int(offset[0] / scale), int(offset[1] / scale)]
        offset_test_thumbnail = Image.fromarray(np.array(template_thumbnail))
        offset_test_thumbnail.paste(test_thumbnail, thumb_offset)

        error_img = Image.fromarray(
            np.array(template_thumbnail.convert("L")) - np.array(offset_test_thumbnail.convert("L")))

        fig = plt.figure()
        ax1 = fig.add_subplot(221)
        ax1.imshow(template_thumbnail)
        ax2 = fig.add_subplot(222)
        ax2.imshow(test_thumbnail)
        ax3 = fig.add_subplot(223)
        ax3.imshow(offset_test_thumbnail)
        ax4 = fig.add_subplot(224)
        ax4.imshow(error_img, cmap="gray")
        plt.show()

    # -----------------------Another way to get offset----------------------- #
    @staticmethod
    def load_QuPath_points_from_QuPath_zip(anno_dir, case_uuid, ext='.tiff'):
        txt_file = os.path.join(anno_dir, case_uuid + ext + "-points", "Points 1.txt")
        fp = open(txt_file, 'r')
        lines = fp.readlines()
        coords_str_list = lines[3:]
        coords = []
        for coord_str in coords_str_list:
            ele = coord_str.strip().split("\t")
            coords.append([ele[0], ele[1]])
        fp.close()
        return np.array(coords).astype(float)

    def get_QuPath_offset_barycentric(self, anno_dir, fixed_case_uuid, float_case_uuid):
        template_anno_points = self.load_QuPath_points_from_QuPath_zip(anno_dir, fixed_case_uuid)
        test_anno_points = self.load_QuPath_points_from_QuPath_zip(anno_dir, float_case_uuid)
        offsets = np.mean(template_anno_points, axis=0) - np.mean(test_anno_points, axis=0)
        return offsets

    # ----------------------- Way to get cell locations----------------------- #
    # attr_filter, please refer to: https://docs.python.org/2/library/xml.etree.elementtree.html
    @staticmethod
    def get_cell_points_from_xml(xml_fn, geo_shape='Points', cell_label_text='mitosis'):
        xml = minidom.parse(xml_fn)
        regions = xml.getElementsByTagName("Region")
        points = np.empty([0, 2], dtype=np.float)
        for region in regions:
            vertices = region.getElementsByTagName("Vertex")
            label_text = region.getAttribute('Text')
            region_geo_shape = region.getAttribute('GeoShape')
            if region_geo_shape == geo_shape and label_text == label_text:
                # Store x, y coordinates into a 2D array in format [x1, y1], [x2, y2], ...
                coords = np.zeros((len(vertices), 2))
                for i, vertex in enumerate(vertices):
                    coords[i][0] = vertex.attributes['X'].value
                    coords[i][1] = vertex.attributes['Y'].value
                points = np.vstack([points, coords])
        return points


# example
if __name__ == "__main__":
    fixed_xml_fn = "/projects/shart/digital_pathology/data/PenMarking/annotations/temp/1c2d01bbee8a41e28357b5ac50b0f5ab.xml"
    float_xml_fn = "/projects/shart/digital_pathology/data/PenMarking/annotations/temp/1c2d01bbee8a41e28357b5ac50b0f5ab.xml"
    offset = OffsetAnnotation().get_xml_offset_barycentric(fixed_xml_fn, float_xml_fn)
    print(offset)

    points_from_fixed = OffsetAnnotation().load_QuPath_points_from_xml(fixed_xml_fn)
    points_from_float = OffsetAnnotation().load_QuPath_points_from_xml(float_xml_fn)
    points_from_float[:, 0] += 800
    points_from_float[:, 1] += 500

    A, t = OffsetAnnotation().get_affine_matrix(points_from_fixed, points_from_float)
    Rotation = math.asin(A[0][1])
    print("Affine transformation matrix:\n", A)
    print("Rotation angle: %.4f" % Rotation)
    print("Affine transformation translation vector:\n", t)
    print("Float image shifting offset: (%.2f %.2f)" % (t[0], t[1]))

