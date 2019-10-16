import openslide
import numpy as np
import os
from skimage.color import rgb2lab
import logging
import tensorflow as tf

class ExtractorParameters:
    """
    Class for establishing & validating parameters for patch extraction
    """
    def __init__(self,  save_dir=None, save_format=".tfrecords", sample_cnt=-1, patch_filter_by_area=None,\
                 with_anno=True, rescale_rate=128, patch_size=128, extract_layer=0):
        if save_dir is None:    # specify a directory to save the extracted patches
            raise Exception("Must specify a directory to save the extraction")
        self.save_dir = save_dir                # Output dir
        self.save_format = save_format          # Save to tfrecords or jpg
        self.with_anno = with_anno              # If true, you need to supply an additional XML file
        self.rescale_rate = rescale_rate        # Fold size to scale the thumbnail to (for faster processing)
        self.patch_size = patch_size            # Size of patches to extract (Height & Width)
        self.extract_layer = extract_layer      # OpenSlide Level
        self.patch_filter_by_area = patch_filter_by_area # Amount of tissue that should be present in a patch
        self.sample_cnt = sample_cnt            # Limit the number of patches to extract (-1 == all patches)


class PatchExtractor:
    """
    Class that sets up the remaining info for patch extraction, and contains the function to extract them
    """

    def __init__(self, detector, parameters,
                 feature_map=None,  # See note below
                 annotations=None   # Object of Annotation Class (see other note below)
                ):
        self.tissue_detector = detector
        self.save_dir = parameters.save_dir
        self.rescale_rate = parameters.rescale_rate     # Fold size to scale the thumbnail to (for faster processing)
        self.patch_size = parameters.patch_size         # Size of patches to extract (Height & Width)
        self.extract_layer = parameters.extract_layer   # OpenSlide Level
        self.save_format = parameters.save_format       # Save to tfrecords or jpg
        self.patch_filter_by_area = parameters.patch_filter_by_area # Amount of tissue that should be present in a patch
        self.sample_cnt = parameters.sample_cnt         # Limit the number of patches to extract (-1 == all patches)
        self.feature_map = feature_map                  # Instructions for building tfrecords
        self.annotations = annotations                  # Annotation object
        if self.save_format == ".tfrecords":
            if feature_map is not None:
                self.with_feature_map = True
            else:  # feature map for tfrecords, if save_format is ".tfrecords", it can't be None
                raise Exception("A Feature map must be specified when you create tfrecords")
        else:
            if feature_map is not None:
                logging.info("No need to specify feature_map ... ignoring.")
            self.with_feature_map = False
        if annotations is None:
            self.with_anno = False
        else:
            self.with_anno = True  # extract with annotation or not

    @staticmethod
    def get_case_info(wsi_fn):
        """
        Converts the WSI filename into an OpenSlideObject and returns it and a dictionary of sample details
        :param wsi_fn: Name of WSI file
        :return: OpenSlideObject, case_description.dict
        """
        wsi_obj = openslide.open_slide(wsi_fn)
        root_dir, fn = os.path.split(wsi_fn)
        uuid, ext = os.path.splitext(fn)
        case_info = {"fn_str": uuid, "ext": ext, "root_dir": root_dir}  # TODO: get file information from the file name
        return wsi_obj, case_info

    def get_thumbnail(self, wsi_obj):
        """
        Given an OpenSlideObject, return a down-sampled thumbnail image
        :param wsi_obj: OpenSlideObject
        :return: thumbnail_image
        """
        wsi_w, wsi_h = wsi_obj.dimensions
        thumb_size_x = wsi_w / self.rescale_rate
        thumb_size_y = wsi_h / self.rescale_rate
        thumbnail = wsi_obj.get_thumbnail([thumb_size_x, thumb_size_y]).convert("RGB")
        return thumbnail

    def get_patch_locations(self, wsi_thumb_mask):
        """
        Given a binary mask representing the thumbnail image,  either return all the pixel positions that are positive,
        or a limited number of pixels that are positive

        :param wsi_thumb_mask: binary mask image with 1 for yes and 0 for no
        :return: coordinate array where the positive pixels are
        """
        pos_indices = np.where(wsi_thumb_mask > 0)
        if self.sample_cnt == -1:  # sample all the image patches
            loc_y = (np.array(pos_indices[0]) * self.rescale_rate).astype(np.int)
            loc_x = (np.array(pos_indices[1]) * self.rescale_rate).astype(np.int)
        else:
            xy_idx = np.random.choice(pos_indices[0].shape[0], self.sample_cnt)
            loc_y = np.array(pos_indices[0][xy_idx] * self.rescale_rate).astype(np.int)
            loc_x = np.array(pos_indices[1][xy_idx] * self.rescale_rate).astype(np.int)
        return [loc_x, loc_y]

    @staticmethod
    def filter_by_content_area(rgb_image_array, area_threshold=0.4, brightness=85):
        """
        Takes an RGB image array as input,
            converts into LAB space
            checks whether the brightness value exceeds the threshold
            returns a boolean indicating whether the amount of tissue > minimum required

        :param rgb_image_array:
        :param area_threshold:
        :param brightness:
        :return:
        """
        # TODO: Alternative tissue detectors, not just RGB->LAB->Thresh
        rgb_image_array[np.any(rgb_image_array == [0, 0, 0], axis=-1)] = [255, 255, 255]
        lab_img = rgb2lab(rgb_image_array)
        l_img = lab_img[:, :, 0]
        binary_img = l_img < brightness
        tissue_size = np.where(binary_img > 0)[0].size
        tissue_ratio = tissue_size * 3 / rgb_image_array.size  # 3 channels
        if tissue_ratio > area_threshold:
            return True
        else:
            return False

    @staticmethod
    def get_patch_label(patch_loc, annotations):
        # get label txt and id for a patch from annotation
        logging.warning('The get_patch_label function is not yet in operation!!')
        logging.warning('Using default: ("label_txt", 0)')
        label_info = ("label_txt", 0)  # TODO: get_patch_label not implemented
        return label_info

    def generate_patch_fn(self, case_info, patch_loc, label_info=None):
        """
        Creates the filenames

        :param case_info: likely a UUID or sample name
        :param patch_loc: tuple of (x, y) locations for where the patch came from
        :param label_info: #TODO: Need to define this
        :return: outputFileName
        """
        if label_info is None:
            tmp = (case_info["fn_str"] + "_%d_%d_" + self.save_format) % (int(patch_loc[0]), int(patch_loc[1]))
            fn = os.path.join(self.save_dir, tmp)
        else:
            fn = "temp"+self.save_format
            logging.warning('The generate_patch_fn function is not yet in operation!!')
            logging.warning('Using default: "temp"+self.save_format') # TODO: generate_patch_fn not implemented
        return fn

    def generate_tfrecords_fp(self, case_info):
        """
        Generates the TFRecord filename and writer object
        :param case_info: likely a UUID or sample name
        :return: TFWriterObject, outputFileName
        """
        tmp = case_info["fn_str"] + self.save_format
        fn = os.path.join(self.save_dir, tmp)
        writer = tf.python_io.TFRecordWriter(fn)  # generate tfRecord file handle
        return writer, fn

    # get image patches and write to files
    def save_patch_without_annotation(self, wsi_obj, case_info, indices):
        """
        Saves images in either JPEG, PNG, or TFRecord format and returns the nubmer of patches it saved

        :param wsi_obj: OpenSlideObject
        :param case_info: likely a UUID or sample name
        :param indices: tuple of (x, y) locations for where the patch will come from
        :return: Number of patches written
        """
        patch_cnt = 0
        if self.with_feature_map:
            tf_writer, tf_fn = self.generate_tfrecords_fp(case_info)
        [loc_x, loc_y] = indices
        for idx, lx in enumerate(loc_x):
            patch = wsi_obj.read_region((loc_x[idx], loc_y[idx]),
                                        self.extract_layer,
                                        (self.patch_size, self.patch_size)
                                        ).convert("RGB")

            # Only print out the patches that contain tissue in them (e.g. Content Rich)
            Content_rich = True
            if self.patch_filter_by_area:  # if we need to filter the image patch
                Content_rich = self.filter_by_content_area(np.array(patch), area_threshold=patch_filter_by_area)
            if Content_rich:
                patch_cnt += 1
                if self.with_feature_map:  # Append data to tfRecord file
                    # TODO: maybe need to find another way to do this
                    values = []
                    for eval_str in self.feature_map.eval_str:
                        values.append(eval(eval_str))
                    features = self.feature_map.update_feature_map_eval(values)
                    example = tf.train.Example(features=tf.train.Features(feature=features))  # Create an example protocol buffer
                    tf_writer.write(example.SerializeToString())  # Serialize to string and write on the file
                else:  # save patch to jpg, with label text and id in file name
                    # if logging.DEBUG == logging.root.level:
                    #     import matplotlib.pyplot as plt
                    #     plt.figure(1)
                    #     plt.imshow(patch)
                    #     plt.show()
                    fn = self.generate_patch_fn(case_info, (loc_x[idx], loc_y[idx]))
                    if self.save_format == ".jpg":
                        patch.save(fn)
                    elif self.save_format == ".png":
                        patch.convert("RGBA").save(fn)
                    else:
                        raise Exception("Can't recognize save format")
            else:
                logging.debug("No content found in image patch x: {} y: {}".format(loc_x[idx], loc_y[idx]))
        tf_writer.close()
        return patch_cnt

    def extract(self, wsi_fn):
        """
        Extract image patches

        :param wsi_fn: a single filename of a WSI
        :return: Number of patches written
        """
        wsi_obj, case_info = self.get_case_info(wsi_fn)
        wsi_thumb = self.get_thumbnail(wsi_obj)    # get the thumbnail
        wsi_thumb_mask = self.tissue_detector.predict(wsi_thumb)   # get the foreground thumbnail mask
        # if logging.DEBUG == logging.root.level:
        #     import matplotlib.pyplot as plt
        #     fig, ax = plt.subplots(2, 1)
        #     ax[0].imshow(wsi_thumb)
        #     ax[1].imshow(wsi_thumb_mask, cmap='gray')
        #     plt.show()
        if not self.with_anno:
            return self.save_patch_without_annotation(wsi_obj, case_info, self.get_patch_locations(wsi_thumb_mask))
        else:
            raise Exception("Saving patches with annotations is not supported yet.")
            # TODO: extract patches with annotations

