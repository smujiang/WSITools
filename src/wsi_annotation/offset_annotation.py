import os
import numpy as np
import openslide
from PIL import Image
import matplotlib.pyplot as plt


class OffsetAnnotation:
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
        # print(coords)
        fp.close()
        return np.array(coords).astype(float)

    def get_anno_offset_barycentric(self, anno_dir, template_case_uuid, test_case_uuid):
        template_anno_points = self.load_QuPath_points_from_QuPath_zip(anno_dir, template_case_uuid)
        test_anno_points = self.load_QuPath_points_from_QuPath_zip(anno_dir, test_case_uuid)
        offsets = np.mean(template_anno_points, axis=0) - np.mean(test_anno_points, axis=0)
        return offsets

    @staticmethod
    def validate_offset(template_wsi_fn, test_wsi_fn, offset, scale=100):
        template_wsi = openslide.open_slide(template_wsi_fn)
        WSI_Width, WSI_Height = template_wsi.dimensions
        thumb_size_x = int(WSI_Width / scale)
        thumb_size_y = int(WSI_Height / scale)
        template_thumbnail = template_wsi.get_thumbnail([thumb_size_x, thumb_size_y])

        test_wsi = openslide.open_slide(test_wsi_fn)
        WSI_Width, WSI_Height = test_wsi.dimensions
        thumb_size_xx = int(WSI_Width / scale)
        thumb_size_yy = int(WSI_Height / scale)
        test_thumbnail = test_wsi.get_thumbnail([thumb_size_xx, thumb_size_yy])

        thumb_offset = [int(offset[0] / scale), int(offset[1] / scale)]
        offset_test_thumbnail = Image.fromarray(np.array(template_thumbnail))
        offset_test_thumbnail.paste(test_thumbnail, thumb_offset)

        error_img = Image.fromarray(np.array(template_thumbnail.convert("L")) - np.array(offset_test_thumbnail.convert("L")))

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

    def get_affine_matrix(self, template_anno_points, test_anno_points):
        return ""

    # TODO: see also: https://pypi.org/project/affine/

    def recover_homogenous_affine_transformation(p, p_prime):
        '''
        Find the unique homogeneous affine transformation that
        maps a set of 3 points to another set of 3 points in 3D
        space:

            p_prime == np.dot(p, R) + t

        where `R` is an unknown rotation matrix, `t` is an unknown
        translation vector, and `p` and `p_prime` are the original
        and transformed set of points stored as row vectors:

            p       = np.array((p1,       p2,       p3))
            p_prime = np.array((p1_prime, p2_prime, p3_prime))

        The result of this function is an augmented 4-by-4
        matrix `A` that represents this affine transformation:

            np.column_stack((p_prime, (1, 1, 1))) == \
                np.dot(np.column_stack((p, (1, 1, 1))), A)

        Source: https://math.stackexchange.com/a/222170 (robjohn)
        '''

        # construct intermediate matrix
        Q = p[1:] - p[0]
        Q_prime = p_prime[1:] - p_prime[0]

        # calculate rotation matrix
        R = np.dot(np.linalg.inv(np.row_stack((Q, np.cross(*Q)))), np.row_stack((Q_prime, np.cross(*Q_prime))))

        # calculate translation vector
        t = p_prime[0] - np.dot(p[0], R)

        # calculate affine transformation matrix
        return np.column_stack((np.row_stack((R, t)), (0, 0, 0, 1)))