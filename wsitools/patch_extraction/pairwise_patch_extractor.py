import openslide
import numpy as np
import os,sys
from skimage.color import rgb2lab
from PIL import Image
import logging
import tensorflow as tf


class PairwiseExtractorParameters:
    def __init__(self,  save_dir, save_format=".tfrecord", sample_cnt=-1, patch_filter_by_area=True, with_anno=True, rescale_rate=128, patch_size=128, extract_layer=0):
        if save_dir is None:    # specify a directory to save the extracted patches
            raise Exception("Must specify a directory to save the extraction")
        self.save_dir = save_dir
        self.save_format = save_format  # Save to .tfrecord .png or .jpg
        self.with_anno = with_anno  # extract with annotation or not
        self.rescale_rate = rescale_rate  # rescale to get the thumbnail
        self.patch_size = patch_size  # patch numbers per image level
        self.extract_layer = extract_layer  # maximum try at each image level
        self.patch_filter_by_area = patch_filter_by_area
        self.sample_cnt = sample_cnt


class PairwisePatchExtractor:
    def __init__(self, detector, parameters, feature_map=None, annotations=None):
        self.tissue_detector = detector
        self.save_dir = parameters.save_dir
        self.rescale_rate = parameters.rescale_rate  # rescale to get the thumbnail
        self.patch_size = parameters.patch_size  # patch numbers per image level
        self.extract_layer = parameters.extract_layer  # maximum try at each image level
        self.save_format = parameters.save_format  # Save to .tfrecord .png or .jpg
        self.patch_filter_by_area = parameters.patch_filter_by_area
        self.sample_cnt = parameters.sample_cnt
        self.feature_map = feature_map
        self.annotations = annotations
        if self.save_format == ".tfrecord":
            if feature_map is not None:
                self.with_feature_map = True
            else:  # feature map for tfRecords, if save_format is ".tfrecord", it can't be None
                raise Exception("No feature map can refer to create tfRecords")
        else:
            if feature_map is not None:
                logging.debug("No need to specify feature mat")
            self.with_feature_map = False
        if annotations is None:
            self.with_anno = False
        else:
            self.with_anno = True  # extract with annotation or not

    @staticmethod
    def get_case_info(wsi_fn):
        wsi_obj = openslide.open_slide(wsi_fn)
        root_dir, fn = os.path.split(wsi_fn)
        uuid, ext = os.path.splitext(fn)
        case_info = {"fn_str": uuid, "ext": ext, "root_dir": root_dir, "dim": wsi_obj.dimensions}
        return wsi_obj, case_info

    # get thumbnails from both WSIs for tissue detection
    def get_thumbnails(self, fixed_wsi_obj, float_wsi_obj):
        fixed_wsi_w, fixed_wsi_h = fixed_wsi_obj.dimensions
        float_wsi_w, float_wsi_h = float_wsi_obj.dimensions
        thumb_size_x = fixed_wsi_w / self.rescale_rate
        thumb_size_y = fixed_wsi_h / self.rescale_rate
        thumbnail_fixed = fixed_wsi_obj.get_thumbnail([thumb_size_x, thumb_size_y]).convert("RGB")
        thumb_size_x = float_wsi_w / self.rescale_rate
        thumb_size_y = float_wsi_h / self.rescale_rate
        thumbnail_float = float_wsi_obj.get_thumbnail([thumb_size_x, thumb_size_y]).convert("RGB")
        return thumbnail_fixed, thumbnail_float

    def get_patch_locations(self, wsi_thumb_mask):
        # ********************************************************
        # Note x, y coordinate order are reversed
        # ********************************************************
        pos_indices = np.where(wsi_thumb_mask > 0)
        if self.sample_cnt == -1:  # sample all the image patches
            loc_y = (np.array(pos_indices[0]) * self.rescale_rate).astype(np.int)   # row
            loc_x = (np.array(pos_indices[1]) * self.rescale_rate).astype(np.int)   # column
        else:
            xy_idx = np.random.choice(pos_indices[0].shape[0], self.sample_cnt)
            loc_y = np.array(pos_indices[0][xy_idx] * self.rescale_rate).astype(np.int)  # row
            loc_x = np.array(pos_indices[1][xy_idx] * self.rescale_rate).astype(np.int)  # column
        return [loc_x, loc_y]

    @staticmethod
    def filter_by_content_area(rgb_image_array, area_threshold=0.4, brightness=85):
        rgb_image_array[np.any(rgb_image_array == [0, 0, 0], axis=-1)] = [255, 255, 255]
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
        if label_text is None or (not label_text.strip()):
            tmp = (case_info["fn_str"] + "_%d_%d" + self.save_format) % (int(patch_loc[0]), int(patch_loc[1]))
        else:
            fn = "temp"+self.save_format
            # TODO:
            tmp = (case_info["fn_str"] + "_%d_%d_%s" + self.save_format) % (
                int(patch_loc[0]), int(patch_loc[1]), label_text)
            print("TODO: add label to file name")
        return os.path.join(self.save_dir, tmp)

    def generate_tfRecord_fp(self, case_info):
        tmp = case_info["fn_str"] + self.save_format
        fn = os.path.join(self.save_dir, tmp)
        writer = tf.python_io.TFRecordWriter(fn)  # create tfRecord file writer
        return writer, fn

    @staticmethod
    def exclude_patch_out_of_bond(fixed_foreground_indices, offset, patch_size, float_wsi_size):
        fixed_foreground_x_list, fixed_foreground_y_list = fixed_foreground_indices
        selected_x = []
        selected_y = []
        for idx, x in enumerate(fixed_foreground_x_list):
            float_x = x + int(offset[0])
            float_y = fixed_foreground_y_list[idx] + int(offset[1])
            # TODO:  x should multiply rescale factor
            if float_x < 0 or float_y < 0 or (float_x + patch_size > float_wsi_size[0]) or (float_y + patch_size > float_wsi_size[1]):
                pass
            else:
                selected_x.append(x)
                selected_y.append(fixed_foreground_y_list[idx])
        # if logging.DEBUG == logging.root.level:
        #     import matplotlib.pyplot as plt
        #     plt.figure(1)
        #     plt.plot(selected_y, selected_x, 'r.')
        #     plt.gca().invert_yaxis()
        #     plt.show()
        return selected_x, selected_y

    # get image patches and write to files
    def save_patch_without_annotation(self, fixed_wsi_obj, float_wsi_obj, fixed_case_info, offset, indices):
        patch_cnt = 0
        if self.with_feature_map:
            tf_writer, tf_fn = self.generate_tfRecord_fp(fixed_case_info)
        [loc_x, loc_y] = indices
        for idx, ly in enumerate(loc_y):
            fixed_patch = fixed_wsi_obj.read_region((loc_x[idx], ly), self.extract_layer, (self.patch_size, self.patch_size)).convert("RGB")
            float_patch = float_wsi_obj.read_region((int(loc_x[idx]+offset[0]), int(ly+offset[1])), self.extract_layer, (self.patch_size, self.patch_size)).convert("RGB")
            # if logging.DEBUG == logging.root.level:
            #     import matplotlib.pyplot as plt
            #     fig, ax = plt.subplots(2, 1)
            #     ax[0].plot(loc_x, loc_y, 'g.')
            #     ax[0].plot(loc_x[idx], ly, 'r.')
            #     ax[0].set_xlim([0, fixed_wsi_obj.dimensions[0]])
            #     ax[0].set_ylim([0, fixed_wsi_obj.dimensions[1]])
            #     ax[0].invert_yaxis()
            #     ax[1].plot(np.array(loc_x)+offset[0], np.array(loc_y)+offset[1], 'g.')
            #     ax[1].plot(int(loc_x[idx]+offset[0]), int(ly+offset[1]), 'bo')
            #     ax[1].set_xlim([0, float_wsi_obj.dimensions[0]])
            #     ax[1].set_ylim([0, float_wsi_obj.dimensions[1]])
            #     ax[1].invert_yaxis()
            #     plt.show()
            Content_rich = True
            # if logging.DEBUG == logging.root.level:
            #     import matplotlib.pyplot as plt
            #     fig, ax = plt.subplots(2, 1)
            #     ax[0].imshow(fixed_patch)
            #     ax[1].imshow(float_patch)
            #     plt.show()
            if self.patch_filter_by_area:  # if we need to filter the image patch
                Content_rich = self.filter_by_content_area(np.array(fixed_patch), area_threshold=0.5) and \
                               self.filter_by_content_area(np.array(float_patch), area_threshold=0.5)
            if Content_rich:
                patch_cnt += 1
                logging.debug("extract from fixe image: %d %d and float image: %d %d" % (loc_x[idx], ly, int(loc_x[idx] + offset[0]), int(ly + offset[1])))
                if self.with_feature_map:  # Append patch to tfRecord file
                    # TODO:
                    # print("Append patch to tfRecord file %s" % tf_fn)
                    values = []
                    for eval_str in self.feature_map.eval_str:
                        values.append(eval(eval_str))
                    features = self.feature_map.update_feature_map_eval(values)
                    example = tf.train.Example(
                        features=tf.train.Features(feature=features))  # Create an example protocol buffer
                    tf_writer.write(example.SerializeToString())  # Serialize to string and write on the file
                else:  # save patch to jpg, with label text and id in file name
                    fn = self.generate_patch_fn(fixed_case_info, (loc_x[idx], ly))
                    fixed_patch_arr = np.array(fixed_patch)
                    # fixed_patch_arr[np.any(fixed_patch_arr == [0, 0, 0], axis=-1)] = [255, 255, 255]  # set black background to white
                    float_patch_arr = np.array(float_patch)
                    # float_patch_arr[np.any(float_patch_arr == [0, 0, 0], axis=-1)] = [255, 255, 255]  # set black background to white
                    comb_arr = np.concatenate([fixed_patch_arr[:, :, :3], float_patch_arr[:, :, :3]], axis=1)
                    if self.save_format == ".jpg":
                        Image.fromarray(comb_arr, 'RGB').save(fn)
                    elif self.save_format == ".png":
                        Image.fromarray(comb_arr, 'RGB').convert("RGBA").save(fn)
                    else:
                        raise Exception("Can't recognize save format")
            else:
                logging.debug("Ignore the patch")
        return patch_cnt

    # get image patches and write to files
    def save_patch_pairs(self, fixed_wsi_obj, float_wsi_obj, fixed_case_info, offset, indices):
        patch_cnt = 0
        if self.with_feature_map:
            tf_writer, tf_fn = self.generate_tfRecord_fp(fixed_case_info)
        [loc_x, loc_y] = indices
        for idx, ly in enumerate(loc_y):
            fixed_patch = fixed_wsi_obj.read_region((loc_x[idx], ly), self.extract_layer, (self.patch_size, self.patch_size)).convert("RGB")
            float_patch = float_wsi_obj.read_region((int(loc_x[idx] + offset[0]), int(ly + offset[1])), self.extract_layer, (self.patch_size, self.patch_size)).convert("RGB")
            Content_rich = True
            if self.patch_filter_by_area:  # if we need to filter the image patch
                Content_rich = self.filter_by_content_area(np.array(fixed_patch), area_threshold=0.5) and \
                               self.filter_by_content_area(np.array(float_patch), area_threshold=0.5)
            if Content_rich:
                patch_cnt += 1
                if self.with_anno:
                    label_id, label_txt = self.get_patch_label([loc_x[idx], loc_y[idx]])
                else:
                    label_txt = ""
                    label_id = -1  # can't delete this line, it will be used if save patch into tfRecords
                logging.debug("extract from fixe image: %d %d and float image: %d %d" % (
                loc_x[idx], ly, int(loc_x[idx] + offset[0]), int(ly + offset[1])))
                if self.with_feature_map:  # Append patch to tfRecord file
                    # TODO:
                    # print("Append patch to tfRecord file %s" % tf_fn)
                    values = []
                    for eval_str in self.feature_map.eval_str:
                        values.append(eval(eval_str))
                    features = self.feature_map.update_feature_map_eval(values)
                    example = tf.train.Example(
                        features=tf.train.Features(feature=features))  # Create an example protocol buffer
                    tf_writer.write(example.SerializeToString())  # Serialize to string and write on the file
                else:  # save patch to jpg, with label text and id in file name
                    fn = self.generate_patch_fn(fixed_case_info, (loc_x[idx], ly), label_text=label_txt)
                    fixed_patch_arr = np.array(fixed_patch)
                    # fixed_patch_arr[np.any(fixed_patch_arr == [0, 0, 0], axis=-1)] = [255, 255, 255]  # set black background to white
                    float_patch_arr = np.array(float_patch)
                    # float_patch_arr[np.any(float_patch_arr == [0, 0, 0], axis=-1)] = [255, 255, 255]  # set black background to white
                    comb_arr = np.concatenate([fixed_patch_arr[:, :, :3], float_patch_arr[:, :, :3]], axis=1)
                    if self.save_format == ".jpg":
                        Image.fromarray(comb_arr, 'RGB').save(fn)
                    elif self.save_format == ".png":
                        Image.fromarray(comb_arr, 'RGB').convert("RGBA").save(fn)
                    else:
                        raise Exception("Can't recognize save format")
            else:
                logging.debug("Ignore the patch")
        return patch_cnt

    def extract(self, fixed_wsi_fn, float_wsi_fn, offset):
        fixed_wsi_obj, fixed_case_info = self.get_case_info(fixed_wsi_fn)
        float_wsi_obj, float_case_info = self.get_case_info(float_wsi_fn)
        thumbnail_fixed, thumbnail_float = self.get_thumbnails(fixed_wsi_obj, float_wsi_obj)    # get the thumbnail
        fixed_wsi_thumb_mask = self.tissue_detector.predict(thumbnail_fixed)   # get the foreground thumbnail mask
        intersection_indices = self.exclude_patch_out_of_bond(self.get_patch_locations(fixed_wsi_thumb_mask), offset, self.patch_size, float_case_info['dim'])
        if logging.DEBUG == logging.root.level:
            print("%d patches need to be extracted" % len(intersection_indices[0]))
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(3, 1)
            ax[0].imshow(thumbnail_fixed)
            ax[1].imshow(thumbnail_float)
            ax[2].imshow(fixed_wsi_thumb_mask, cmap='gray')
            plt.show()
        return self.save_patch_pairs(fixed_wsi_obj, float_wsi_obj, fixed_case_info, offset, intersection_indices)


        # if logging.DEBUG == logging.root.level:
        #     print("%d patches need to be extracted" % len(intersection_indices[0]))
        #     import matplotlib.pyplot as plt
        #     fig, ax = plt.subplots(3, 1)
        #     ax[0].imshow(thumbnail_fixed)
        #     ax[1].imshow(thumbnail_float)
        #     ax[2].imshow(fixed_wsi_thumb_mask, cmap='gray')
        #     plt.show()
        # if not self.with_anno:
        #     return self.save_patch_without_annotation(fixed_wsi_obj, float_wsi_obj, fixed_case_info, offset, intersection_indices)
        # else:
        #     # TODO:
        #     print("TODO: extract patches with annotations")

    def extract_parallel(self, ffo_tuple):
        fixed_wsi_fn, float_wsi_fn, offset_x, offset_y = ffo_tuple
        offset = (offset_x, offset_y)
        fixed_wsi_obj, fixed_case_info = self.get_case_info(fixed_wsi_fn)
        float_wsi_obj, float_case_info = self.get_case_info(float_wsi_fn)
        thumbnail_fixed, thumbnail_float = self.get_thumbnails(fixed_wsi_obj, float_wsi_obj)    # get the thumbnail
        fixed_wsi_thumb_mask = self.tissue_detector.predict(thumbnail_fixed)   # get the foreground thumbnail mask
        intersection_indices = self.exclude_patch_out_of_bond(self.get_patch_locations(fixed_wsi_thumb_mask), offset, self.patch_size, float_case_info['dim'])
        if logging.DEBUG == logging.root.level:
            print("%d patches need to be extracted" % len(intersection_indices[0]))
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(3, 1)
            ax[0].imshow(thumbnail_fixed)
            ax[1].imshow(thumbnail_float)
            ax[2].imshow(fixed_wsi_thumb_mask, cmap='gray')
            plt.show()
        if not self.with_anno:
            return self.save_patch_without_annotation(fixed_wsi_obj, float_wsi_obj, fixed_case_info, offset, intersection_indices)
        else:
            # TODO:
            print("TODO: extract patches with annotations")

# example
if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.DEBUG)
    # logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.ERROR)

    # fixed_wsi = "/projects/shart/digital_pathology/data/PenMarking/WSIs/MELF/7bb50b5d9dcf4e53ad311d66136ae00f.tiff"
    # float_wsi = "/projects/shart/digital_pathology/data/PenMarking/WSIs/MELF-Clean/8a26a55a78b947059da4e8c36709a828.tiff"
    # fixed_wsi = "/projects/shart/digital_pathology/data/PenMarking/WSIs/MELF/d83cc7d1c941438e93786fc381ab5bb5.tiff"
    # fixed_wsi = "/projects/shart/digital_pathology/data/PenMarking/WSIs/MELF/e39a8d60a56844d695e9579bce8f0335.tiff"
    # float_wsi_root_dir = "/projects/shart/digital_pathology/data/PenMarking/WSIs/MELF-Clean"
    # gnb_training_files = "/projects/shart/digital_pathology/data/PenMarking/model/tissue_loc/HE_tissue_others.tsv"
    #
    # from wsitools.file_management.wsi_case_manager import WSI_CaseManager  # # import dependent packages
    # from wsitools.file_management.offset_csv_manager import OffsetCSVManager
    # from wsitools.tissue_detection.tissue_detector import TissueDetector
    #
    # tissue_detector = TissueDetector("GNB", threshold=0.5, training_files=gnb_training_files)
    #
    # case_mn = WSI_CaseManager()
    # float_wsi = case_mn.get_counterpart_fn(fixed_wsi, float_wsi_root_dir)
    # _, fixed_wsi_uuid, _ = case_mn.get_wsi_fn_info(fixed_wsi)
    # _, float_wsi_uuid, _ = case_mn.get_wsi_fn_info(float_wsi)
    # # offset_csv_fn = "/projects/shart/digital_pathology/data/PenMarking/WSIs/registration_offsets.csv"
    # offset_csv_fn = "../file_management/example/wsi_pair_offset.csv"
    #
    # offset_csv_mn = OffsetCSVManager(offset_csv_fn)
    # offset, state_indicator = offset_csv_mn.lookup_table(fixed_wsi_uuid, float_wsi_uuid)
    # if state_indicator == 0:
    #     raise Exception("No corresponding offset can be found in the file")
    #
    # # extract pairs of patches without annotation, no feature map specified and save patches to '.jpg'
    # output_dir = "/projects/shart/digital_pathology/data/PenMarking/temp"
    # parameters = PairwiseExtractorParameters(output_dir, save_format='.jpg', sample_cnt=-1)
    # patch_extractor = PairwisePatchExtractor(tissue_detector, parameters, feature_map=None, annotations=None)
    # patch_cnt = patch_extractor.extract(fixed_wsi, float_wsi, offset)
    # print("%d Patches have been save to %s" % (patch_cnt, output_dir))

    # # multiple processing
    # from wsitools.file_management.wsi_case_manager import WSI_CaseManager  # # import dependent packages
    # from wsitools.file_management.offset_csv_manager import OffsetCSVManager
    # from wsitools.file_management.case_list_manager import CaseListManager
    # from wsitools.tissue_detection.tissue_detector import TissueDetector
    # import multiprocessing
    #
    # float_wsi_root_dir = "/projects/shart/digital_pathology/data/PenMarking/WSIs/MELF-Clean"
    #
    # gnb_training_files = "/projects/shart/digital_pathology/data/PenMarking/model/tissue_loc/HE_tissue_others.tsv"
    # tissue_detector = TissueDetector("GNB", threshold=0.5, training_files=gnb_training_files)
    #
    # offset_csv_fn = "/projects/shart/digital_pathology/data/PenMarking/WSIs/registration_offsets.csv"
    # offset_csv_mn = OffsetCSVManager(offset_csv_fn)
    #
    # case_list_txt = "/projects/shart/digital_pathology/data/PenMarking/WSIs/annotated_cases.txt"
    # case_list_mn = CaseListManager(case_list_txt)
    # all_fixed_wsi_fn = case_list_mn.case_list
    #
    # case_pair_mn = WSI_CaseManager()
    #
    # all_fixed_float_offset = []
    # for fixed_wsi in all_fixed_wsi_fn:
    #     float_wsi = case_pair_mn.get_counterpart_fn(fixed_wsi, float_wsi_root_dir)
    #     _, fixed_wsi_uuid, _ = case_pair_mn.get_wsi_fn_info(fixed_wsi)
    #     _, float_wsi_uuid, _ = case_pair_mn.get_wsi_fn_info(float_wsi)
    #
    #     offset, state_indicator = offset_csv_mn.lookup_table(fixed_wsi_uuid, float_wsi_uuid)
    #     if state_indicator == 0:
    #         raise Exception("No corresponding offset can be found in the file")
    #     all_fixed_float_offset.append((fixed_wsi, float_wsi, offset[0], offset[1]))
    #
    # # extract pairs of patches without annotation, no feature map specified and save patches to '.jpg'
    # output_dir = "/projects/shart/digital_pathology/data/PenMarking/temp"
    # parameters = PairwiseExtractorParameters(output_dir, save_format='.jpg', sample_cnt=-1)
    # patch_extractor = PairwisePatchExtractor(tissue_detector, parameters, feature_map=None, annotations=None)
    #
    # multiprocessing.set_start_method('spawn')
    # pool = multiprocessing.Pool(processes=4)
    # pool.map(patch_extractor.extract_parallel, all_fixed_float_offset)

    #  # Save into tfRecords
    from wsitools.file_management.wsi_case_manager import WSI_CaseManager  # # import dependent packages
    from wsitools.file_management.offset_csv_manager import OffsetCSVManager
    from wsitools.tissue_detection.tissue_detector import TissueDetector
    from wsitools.patch_extraction.feature_map_creator import FeatureMapCreator

    fixed_wsi = "/projects/shart/digital_pathology/data/PenMarking/WSIs/MELF/7bb50b5d9dcf4e53ad311d66136ae00f.tiff"
    float_wsi_root_dir = "/projects/shart/digital_pathology/data/PenMarking/WSIs/MELF-Clean"

    gnb_training_files = "/projects/shart/digital_pathology/data/PenMarking/model/tissue_loc/HE_tissue_others.tsv"
    tissue_detector = TissueDetector("GNB", threshold=0.5, training_files=gnb_training_files)

    offset_csv_fn = "/projects/shart/digital_pathology/data/PenMarking/WSIs/registration_offsets.csv"
    offset_csv_mn = OffsetCSVManager(offset_csv_fn)

    fm = FeatureMapCreator("./feature_maps/basic_fm_PP_eval.csv")

    case_mn = WSI_CaseManager()
    float_wsi = case_mn.get_counterpart_fn(fixed_wsi, float_wsi_root_dir)
    _, fixed_wsi_uuid, _ = case_mn.get_wsi_fn_info(fixed_wsi)
    _, float_wsi_uuid, _ = case_mn.get_wsi_fn_info(float_wsi)

    offset, state_indicator = offset_csv_mn.lookup_table(fixed_wsi_uuid, float_wsi_uuid)
    if state_indicator == 0:
        raise Exception("No corresponding offset can be found in the file")

    output_dir = "/projects/shart/digital_pathology/data/PenMarking/temp"
    parameters = PairwiseExtractorParameters(output_dir, save_format='.tfrecord', sample_cnt=-1)
    patch_extractor = PairwisePatchExtractor(tissue_detector, parameters, feature_map=fm, annotations=None)
    patch_cnt = patch_extractor.extract(fixed_wsi, float_wsi, offset)
    print("%d Patches have been save to %s" % (patch_cnt, output_dir))









