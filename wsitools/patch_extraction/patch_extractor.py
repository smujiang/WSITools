import openslide
import numpy as np
import os
from skimage.color import rgb2lab
import logging
import tensorflow as tf
import sys
import concurrent  # python 2.7 don't support this module
from PIL import Image, ImageDraw
from scipy import ndimage

logger = logging.getLogger(__name__)
ch = logging.StreamHandler()
formatter = logging.Formatter('\x1b[80D\x1b[1A\x1b[K%(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)
patch_cnt = 0  # count how many patches extracted


class ExtractorParameters:
    """
    Class for establishing & validating parameters for patch extraction
    """

    def __init__(self, save_dir=None, log_dir=None, save_format=".tfrecord", sample_cnt=-1, patch_filter_by_area=None, \
                 with_anno=True, threads=20, rescale_rate=128, patch_size=128, stride=128, patch_rescale_to=None,
                 extract_layer=0):
        if save_dir is None:  # specify a directory to save the extracted patches
            raise Exception("Must specify a directory to save the extraction")
        self.save_dir = save_dir  # Output dir
        self.log_dir = log_dir  # Output dir
        self.save_format = save_format  # Save to .tfrecord or .jpg
        self.with_anno = with_anno  # If true, you need to supply an additional XML file
        self.rescale_rate = rescale_rate  # Fold size to scale the thumbnail to (for faster processing)
        self.patch_size = patch_size  # Size of patches to extract (Height & Width)
        self.stride = stride  # stride for patch extraction
        self.patch_rescale_to = patch_rescale_to  # rescale the extracted patches
        self.extract_layer = extract_layer  # OpenSlide Level
        self.patch_filter_by_area = patch_filter_by_area  # Amount of tissue that should be present in a patch
        self.sample_cnt = sample_cnt  # Limit the number of patches to extract (-1 == all patches)
        self.threads = threads


class PatchExtractor:
    """
    Class that sets up the remaining info for patch extraction, and contains the function to extract them
    """

    def __init__(self, detector=None, parameters=None,
                 feature_map=None,  # See note below
                 annotations=None  # Object of Annotation Class (see other note below)
                 ):
        self.tissue_detector = detector
        self.threads = parameters.threads
        self.save_dir = parameters.save_dir
        self.log_dir = parameters.log_dir
        self.rescale_rate = parameters.rescale_rate  # Fold size to scale the thumbnail to (for faster processing)
        self.patch_size = parameters.patch_size  # Size of patches to extract (Height & Width)
        self.stride = parameters.stride  # stride for patch extraction
        self.patch_rescale_to = parameters.patch_rescale_to  # rescale the extracted patches
        self.extract_layer = parameters.extract_layer  # OpenSlide Level
        self.save_format = parameters.save_format  # Save to .tfrecord or .jpg
        self.patch_filter_by_area = parameters.patch_filter_by_area  # Amount of tissue that should be present in a patch
        self.sample_cnt = parameters.sample_cnt  # Limit the number of patches to extract (-1 == all patches)
        self.feature_map = feature_map  # Instructions for building tfRecords
        self.annotations = annotations  # Annotation object
        if self.save_format == ".tfrecord":
            if feature_map is not None:
                self.with_feature_map = True
            else:  # feature map for tfRecords, if save_format is ".tfrecord", it can't be None
                raise Exception("A Feature map must be specified when you create tfRecords")
        else:
            if feature_map is not None:
                logger.info("No need to specify feature_map ... ignoring.")
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

    def get_patch_locations(self, wsi_thumb_mask, level_downsamples):
        """
        Given a binary mask representing the thumbnail image,  either return all the pixel positions that are positive,
        or a limited number of pixels that are positive

        :param wsi_thumb_mask: binary mask image with 1 for yes and 0 for no
        :return: coordinate array where the positive pixels are
        """

        wsi_thumb_mask = ndimage.binary_erosion(wsi_thumb_mask)
        pos_indices = np.where(wsi_thumb_mask > 0)
        loc_y = (np.array(pos_indices[0]) * self.rescale_rate).astype(np.int)
        loc_x = (np.array(pos_indices[1]) * self.rescale_rate).astype(np.int)
        loc_x_selected = []
        loc_y_selected = []
        x_lim = [min(loc_x), max(loc_x)]
        y_lim = [min(loc_y), max(loc_y)]
        for x in range(x_lim[0], x_lim[1], int(self.stride*level_downsamples[self.extract_layer])):
            for y in range(y_lim[0], y_lim[1], int(self.stride*level_downsamples[self.extract_layer])):
                x_idx = int(x / self.rescale_rate)
                y_idx = int(y / self.rescale_rate)
                x_idx_1 = int((x+self.patch_size * level_downsamples[self.extract_layer]) / self.rescale_rate)
                y_idx_1 = int((y+self.patch_size * level_downsamples[self.extract_layer]) / self.rescale_rate)
                if x_idx_1 >= wsi_thumb_mask.shape[1]:
                    x_idx_1 = x_idx
                if y_idx_1 >= wsi_thumb_mask.shape[0]:
                    y_idx_1 = y_idx
                if np.count_nonzero(wsi_thumb_mask[y_idx:y_idx_1, x_idx:x_idx_1]) > 0:
                    loc_x_selected.append(int(x))
                    loc_y_selected.append(int(y))
        return [loc_x_selected, loc_y_selected]

    def get_patch_locations_from_ROIs(self, ROIs, level_downsamples):
        """
        Given a ROI list,  either return all the pixel positions that are in ROI
        :param ROIs: ROIs [[min_x, min_y, max_x, max_y], ...]
        :return: coordinate array where the positive pixels are
        """
        loc_x_selected = []
        loc_y_selected = []
        for roi in ROIs:
            x_lim = [roi[0], roi[2]]
            y_lim = [roi[1], roi[3]]
            for x in range(x_lim[0], x_lim[1], int(self.stride*level_downsamples[self.extract_layer])):
                for y in range(y_lim[0], y_lim[1], int(self.stride*level_downsamples[self.extract_layer])):
                    loc_x_selected.append(int(x))
                    loc_y_selected.append(int(y))
        return [loc_x_selected, loc_y_selected]

    def validate_extract_locations(self, case_info, locations, thumbnail, level_downsamples):
        """
        create a figure to validate the locations
        :param locations:
        :return:
        """
        if self.log_dir is None:
            print("log dir is None, validation image will not be saved")
            return
        if not os.path.exists(self.log_dir):
            try:
                os.makedirs(self.log_dir,  exist_ok=True)
            except OSError:
                raise Exception("Can't create/access log_dir, unable to save validation image")
        if os.path.exists(self.log_dir):
            draw = ImageDraw.Draw(thumbnail)
            [loc_x_selected, loc_y_selected] = locations
            for i in range(len(loc_x_selected)):
                xy = [int(loc_x_selected[i] / self.rescale_rate),
                      int(loc_y_selected[i] / self.rescale_rate),
                      int((loc_x_selected[i] + self.patch_size * level_downsamples[
                          self.extract_layer]) / self.rescale_rate),
                      int((loc_y_selected[i] + self.patch_size * level_downsamples[
                          self.extract_layer]) / self.rescale_rate)]
                draw.rectangle(xy, outline='green')
            # thumbnail.show()
            print("Grids numbers in total: %d" % len(loc_x_selected))
            thumb_fn = os.path.join(self.log_dir, case_info["fn_str"]+"_extraction_grid_" + str(len(loc_x_selected)) + ".png")
            thumbnail.save(thumb_fn)

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
        # rgb_image_array[np.any(rgb_image_array == [0, 0, 0], axis=-1)] = [255, 255, 255]
        lab_img = rgb2lab(rgb_image_array)
        l_img = lab_img[:, :, 0]
        binary_img_array_1 = np.array(0 < l_img)
        binary_img_array_2 = np.array(l_img < brightness)
        binary_img = np.logical_and(binary_img_array_1, binary_img_array_2) * 255
        tissue_size = np.where(binary_img > 0)[0].size
        tissue_ratio = tissue_size * 3 / rgb_image_array.size  # 3 channels
        if tissue_ratio > area_threshold:
            return True
        else:
            return False

    def get_patch_label(self, patch_loc, Center=True):
        """
        :param patch_loc:  where the patch is extracted(top left)
        :param Center:  use the top left (False) or the center of the patch (True) to get the annotation label
        :return: label ID and label text
        """
        if Center:
            pix_loc = (patch_loc[0] + self.patch_size, patch_loc[1] + self.patch_size)
        else:
            pix_loc = patch_loc
        label_id, label_txt = self.annotations.get_pixel_label(pix_loc)
        return label_id, label_txt

    def generate_patch_fn(self, case_info, patch_loc, label_text=None):
        """
        Creates the filenames, if we save the patches as jpg/png files.

        :param case_info: likely a UUID or sample name
        :param patch_loc: tuple of (x, y) locations for where the patch came from
        :param label_text: #TODO: Need to define this
        :return: outputFileName
        """
        if label_text is None:
            tmp = (case_info["fn_str"] + "_%d_%d" + self.save_format) % (int(patch_loc[0]), int(patch_loc[1]))
        else:
            tmp = (case_info["fn_str"] + "_%d_%d_%s" + self.save_format) % (
                int(patch_loc[0]), int(patch_loc[1]), label_text)
        return os.path.join(self.save_dir, case_info["fn_str"], tmp)

    def generate_tfRecords_fp(self, case_info):
        """
        Generates the TFRecord filename and writer object
        :param case_info: likely a UUID or sample name
        :return: TFWriterObject, outputFileName
        """
        tmp = case_info["fn_str"] + self.save_format
        fn = os.path.join(self.save_dir, tmp)
        writer = tf.io.TFRecordWriter(fn)  # generate tfRecord file handle
        return writer, fn

    def img_patch_generator(self, x, y, wsi_obj, case_info, tf_writer=None):
        """Return image patches if they have enough tissue"""
        patch = wsi_obj.read_region((x, y),
                                    self.extract_layer,
                                    (self.patch_size, self.patch_size)
                                    ).convert("RGB")
        if self.patch_rescale_to:
            patch = patch.resize([self.patch_rescale_to, self.patch_rescale_to])

        # Only print out the patches that contain tissue in them (e.g. Content Rich)
        Content_rich = True
        if self.patch_filter_by_area:  # if we need to filter the image patch
            Content_rich = self.filter_by_content_area(np.array(patch), area_threshold=self.patch_filter_by_area)
        if Content_rich:
            global patch_cnt
            patch_cnt += 1
            if self.with_anno:
                label_id, label_txt = self.get_patch_label([x, y])
            else:
                label_txt = None
                label_id = -1  # can't delete this line, it will be used if save patch into tfRecords

            if self.with_feature_map:  # Append data to tfRecord file
                # TODO: maybe need to find another way to do this
                values = []
                for eval_str in self.feature_map.eval_str:
                    values.append(eval(eval_str))
                features = self.feature_map.update_feature_map_eval(values)
                example = tf.train.Example(
                    features=tf.train.Features(feature=features))  # Create an example protocol buffer
                tf_writer.write(example.SerializeToString())  # Serialize to string and write on the file
                sys.stdout.flush()
            else:  # save patch to jpg, with label text and id in file name
                fn = self.generate_patch_fn(case_info, (x, y), label_text=label_txt)
                if os.path.exists(fn):
                    logger.error('You already wrote this image file')
                if self.save_format == ".jpg":
                    patch.save(fn)
                elif self.save_format == ".png":
                    patch.convert("RGBA").save(fn)
                else:
                    raise Exception("Can't recognize save format")
                sys.stdout.flush()
        else:
            logger.debug("No content found in image patch x: {} y: {}".format(x, y))

    def parallel_save_patches(self, wsi_obj, case_info, indices):
        if self.with_feature_map:
            tf_writer, tf_fn = self.generate_tfRecords_fp(case_info)
        else:
            tf_writer = None
        [loc_x, loc_y] = indices
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.threads) as executor:
            futures = [executor.submit(self.img_patch_generator, loc_x[idx], loc_y[idx], wsi_obj, case_info, tf_writer)
                       for idx, lx
                       in enumerate(loc_x)]
            for f in concurrent.futures.as_completed(futures):
                try:
                    f.result()
                except NameError:
                    # logger.warning('Unable to find x_loc: {}'.format(loc_x))
                    pass
        if self.with_feature_map:
            tf_writer.close()
        global patch_cnt
        logger.info('Found {} image patches'.format(patch_cnt))

    # get image patches and write to files
    def save_patch_without_annotation(self, wsi_obj, case_info, indices):
        """
        Saves images in either JPEG, PNG, or TFRecord format and returns the nubmer of patches it saved

        :param wsi_obj: OpenSlideObject
        :param case_info: likely a UUID or sample name
        :param indices: tuple of (x, y) locations for where the patch will come from
        :param threads: how many threads to use
        :return: Number of patches written
        """

        if self.with_feature_map:
            tf_writer, tf_fn = self.generate_tfRecords_fp(case_info)
        [loc_x, loc_y] = indices
        for idx, lx in enumerate(loc_x):
            patch = wsi_obj.read_region((loc_x[idx], loc_y[idx]),
                                        self.extract_layer,
                                        (self.patch_size, self.patch_size)
                                        ).convert("RGB")

            # Only print out the patches that contain tissue in them (e.g. Content Rich)
            Content_rich = True
            if self.patch_filter_by_area:  # if we need to filter the image patch
                Content_rich = self.filter_by_content_area(np.array(patch), area_threshold=self.patch_filter_by_area)
            if Content_rich:
                patch_cnt += 1
                if self.with_feature_map:  # Append data to tfRecord file
                    # TODO: maybe need to find another way to do this
                    values = []
                    for eval_str in self.feature_map.eval_str:
                        values.append(eval(eval_str))
                    features = self.feature_map.update_feature_map_eval(values)
                    example = tf.train.Example(
                        features=tf.train.Features(feature=features))  # Create an example protocol buffer
                    tf_writer.write(example.SerializeToString())  # Serialize to string and write on the file
                    logger.info('\rWrote {} to tfRecords '.format(patch_cnt))
                    sys.stdout.flush()
                else:  # save patch to jpg, with label text and id in file name
                    # if logger.DEBUG == logger.root.level:
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
                    logger.info('\rWrote {} to image files '.format(patch_cnt))
                    sys.stdout.flush()
            else:
                logger.debug("No content found in image patch x: {} y: {}".format(loc_x[idx], loc_y[idx]))
        tf_writer.close()
        return patch_cnt

    # get image patches and write to files
    def save_patches(self, wsi_obj, case_info, indices):
        """
        Saves images (and their labels) in either JPEG, PNG, or TFRecord format and returns the number of patches it saved

        :param wsi_obj: OpenSlideObject
        :param case_info: likely a UUID or sample name
        :param indices: tuple of (x, y) locations for where the patch will come from
        :return: Number of patches written
        """
        patch_cnt = 0
        if self.with_feature_map:
            tf_writer, tf_fn = self.generate_tfRecords_fp(case_info)
        [loc_x, loc_y] = indices
        for idx, lx in enumerate(loc_x):
            patch = wsi_obj.read_region((loc_x[idx], loc_y[idx]),
                                        self.extract_layer,
                                        (self.patch_size, self.patch_size)
                                        ).convert("RGB")
            if self.patch_rescale_to:
                patch = patch.resize([self.patch_rescale_to, self.patch_rescale_to])
            # Only print out the patches that contain tissue in them (e.g. Content Rich)
            Content_rich = True
            if self.patch_filter_by_area:  # if we need to filter the image patch
                Content_rich = self.filter_by_content_area(np.array(patch), area_threshold=self.patch_filter_by_area)
            if Content_rich:
                patch_cnt += 1
                if self.with_anno:
                    label_id, label_txt = self.get_patch_label([loc_x[idx], loc_y[idx]])
                else:
                    label_txt = None
                    label_id = -1  # can't delete this line, it will be used if save patch into tfRecords
                if self.with_feature_map:  # Append data to tfRecord file
                    # TODO: maybe need to find another way to do this
                    values = []
                    for eval_str in self.feature_map.eval_str:
                        values.append(eval(eval_str))
                    features = self.feature_map.update_feature_map_eval(values)
                    example = tf.train.Example(
                        features=tf.train.Features(feature=features))  # Create an example protocol buffer
                    tf_writer.write(example.SerializeToString())  # Serialize to string and write on the file
                    logger.info('\rWrote {} to tfRecords '.format(patch_cnt))
                    sys.stdout.flush()
                else:  # save patch to jpg, with label text and id in file name
                    # if logger.DEBUG == logger.root.level:
                    #     import matplotlib.pyplot as plt
                    #     plt.figure(1)
                    #     plt.imshow(patch)
                    #     plt.show()
                    fn = self.generate_patch_fn(case_info, (loc_x[idx], loc_y[idx]), label_text=label_txt)
                    if not os.path.exists(os.path.split(fn)[0]):
                        os.makedirs(os.path.split(fn)[0])
                    if self.save_format == ".jpg":
                        patch.save(fn)
                    elif self.save_format == ".png":
                        patch.convert("RGBA").save(fn)
                    else:
                        raise Exception("Can't recognize save format")
                    logger.info('\rWrote {} to image files '.format(patch_cnt))
                    sys.stdout.flush()
                if self.sample_cnt == patch_cnt:
                    if self.with_feature_map:
                        tf_writer.close()
                    return patch_cnt
            else:
                logger.debug("No content found in image patch x: {} y: {}".format(loc_x[idx], loc_y[idx]))
        if self.with_feature_map:
            tf_writer.close()
        return patch_cnt

    def extract(self, wsi_fn):
        """
        Extract image patches from all the foreground(tissue)
        :param wsi_fn: a single filename of a WSI
        :return: Number of patches written
        """
        wsi_obj, case_info = self.get_case_info(wsi_fn)
        wsi_thumb = self.get_thumbnail(wsi_obj)  # get the thumbnail
        wsi_thumb_mask = self.tissue_detector.predict(wsi_thumb)  # get the foreground thumbnail mask
        extract_locations = self.get_patch_locations(wsi_thumb_mask, wsi_obj.level_downsamples)
        self.validate_extract_locations(case_info, extract_locations, wsi_thumb, wsi_obj.level_downsamples)
        return self.save_patches(wsi_obj, case_info, extract_locations)

        # if logger.DEBUG == logger.root.level:
        #     import matplotlib.pyplot as plt
        #     fig, ax = plt.subplots(2, 1)
        #     ax[0].imshow(wsi_thumb)
        #     ax[1].imshow(wsi_thumb_mask, cmap='gray')
        #     plt.show()
        # if not self.with_anno:
        #     return self.save_patch_without_annotation(wsi_obj, case_info, self.get_patch_locations(wsi_thumb_mask))
        # else:
        #     raise Exception("Saving patches with annotations is not supported yet.")

    def extract_ROIs(self, wsi_fn, ROIs):
        '''
        extract patches from ROI list
        :param wsi_fn: WSI file name
        :param ROIs:   example: ROIs = [[35000, 35000, 43000, 43000], [12000, 19000, 25000, 30000]]
        :return:
        '''
        wsi_obj, case_info = self.get_case_info(wsi_fn)
        extract_locations = self.get_patch_locations_from_ROIs(ROIs, wsi_obj.level_downsamples)
        wsi_thumb = self.get_thumbnail(wsi_obj)  # get the thumbnail for validation
        self.validate_extract_locations(case_info, extract_locations, wsi_thumb, wsi_obj.level_downsamples)
        return self.save_patches(wsi_obj, case_info, extract_locations)


if __name__ == "__main__":
    from wsitools.tissue_detection.tissue_detector import TissueDetector  # import dependent packages
    from wsitools.patch_extraction.feature_map_creator import FeatureMapCreator
    from wsitools.wsi_annotation.region_annotation import AnnotationRegions

    # wsi_fn = "/projects/shart/digital_pathology/data/PenMarking/WSIs/MELF/e39a8d60a56844d695e9579bce8f0335.tiff"  # WSI file name
    # output_dir = "/projects/shart/digital_pathology/data/PenMarking/temp"
    #
    # tissue_detector = TissueDetector("LAB_Threshold", threshold=85)  #
    # # fm = FeatureMapCreator("./feature_maps/basic_fm_PL_eval.csv")  # use this template to create feature map
    # # xml_fn = "/projects/shart/digital_pathology/data/PenMarking/annotations/temp/e39a8d60a56844d695e9579bce8f0335.xml"
    # class_label_id_csv = "/projects/shart/digital_pathology/data/PenMarking/annotations/temp/label_id.csv"
    # annotations = AnnotationRegions(xml_fn, class_label_id_csv)
    # parameters = ExtractorParameters(output_dir, save_format='.tfrecord', sample_cnt=-1)
    # patch_extractor = PatchExtractor(tissue_detector, parameters=parameters, feature_map=fm,
    #                                  annotations=annotations)
    # patch_num = patch_extractor.extract(wsi_fn)

    wsi_fn = "\\\\mfad\\researchmn\\HCPR\\HCPR-GYNECOLOGICALTUMORMICROENVIRONMENT\\WSIs\\OCMC-016.svs"  # WSI file name
    output_dir = "H:\\OvarianCancer\\ImageData\\Patches\\OCMC-016"
    log_dir = "H:\\OvarianCancer\\ImageData\\Patches\\OCMC-016_log"
    tissue_detector = TissueDetector("LAB_Threshold", threshold=85)  #

    parameters = ExtractorParameters(output_dir, patch_size=500, stride=500, extract_layer=0, patch_filter_by_area=0.3,
                                      save_format='.jpg', sample_cnt=-1)

    '''
    For example:
    Slide resolution is 40x, but we need 20X image patches (size 512*512)
    WSI level downsamples = [1 , 4, 16, 32], so can't directly read from level 1 to match our requirements
    In our package, we can call the function like below:
    parameters = ExtractorParameters(output_dir, log_dir=log_dir, save_format='.jpg', patch_size=1024, stride=1024, sample_cnt=-1, extract_layer=0, patch_rescale_to=512)
    '''

    patch_extractor = PatchExtractor(tissue_detector, parameters=parameters)
    patch_num = patch_extractor.extract(wsi_fn)
    #
    # ROIs = [[35000, 35000, 43000, 43000], [12000, 19000, 25000, 30000]]  # coordinates are from level 0
    # patch_extractor.extract_ROIs(wsi_fn, ROIs)

    print("%d Patches have been save to %s" % (patch_num, output_dir))
