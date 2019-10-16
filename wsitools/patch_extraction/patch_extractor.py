import openslide
import numpy as np
import os, sys
from skimage.color import rgb2lab
import logging
import tensorflow as tf

class ExtractorParameters:
    def __init__(self,  save_dir, save_format=".tfrecord", sample_cnt=-1, patch_filter_by_area=True, with_anno=True, rescale_rate=128, patch_size=128, extract_layer=0):
        if save_dir is None:    # specify a directory to save the extracted patches
            raise Exception("Must specify a directory to save the extraction")
        self.save_dir = save_dir
        self.save_format = save_format  # Save to tfRecord or jpg
        self.with_anno = with_anno  # extract with annotation or not
        self.rescale_rate = rescale_rate  # rescale to get the thumbnail
        self.patch_size = patch_size  # patch numbers per image level
        self.extract_layer = extract_layer  # maximum try at each image level
        self.patch_filter_by_area = patch_filter_by_area
        self.sample_cnt = sample_cnt


class PatchExtractor:
    def __init__(self, detector, parameters, feature_map=None, annotations=None):
        self.tissue_detector = detector
        self.save_dir = parameters.save_dir
        self.rescale_rate = parameters.rescale_rate  # rescale to get the thumbnail
        self.patch_size = parameters.patch_size  # patch numbers per image level
        self.extract_layer = parameters.extract_layer  # maximum try at each image level
        self.save_format = parameters.save_format  # Save to tfRecord or jpg
        self.patch_filter_by_area = parameters.patch_filter_by_area
        self.sample_cnt = parameters.sample_cnt
        self.feature_map = feature_map
        self.annotations = annotations
        if self.save_format == ".tfRecord":   #tfRecord
            if feature_map is not None:
                self.with_feature_map = True
            else:  # feature map for tfRecords, if save_format is ".tfRecord", it can't be None
                raise Exception("No feature map can refer to create tfRecord")
        else:
            if feature_map is not None:
                logging.info("No need to specify feature mat")
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
        case_info = {"fn_str": uuid, "ext": ext, "root_dir": root_dir}  # TODO: get file information from the file name
        return wsi_obj, case_info

    def get_thumbnail(self, wsi_obj):
        wsi_w, wsi_h = wsi_obj.dimensions
        thumb_size_x = wsi_w / self.rescale_rate
        thumb_size_y = wsi_h / self.rescale_rate
        thumbnail = wsi_obj.get_thumbnail([thumb_size_x, thumb_size_y]).convert("RGB")
        return thumbnail

    def get_patch_locations(self, wsi_thumb_mask):
        print(wsi_thumb_mask.shape)
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
        label_info = ("label_txt", 0)  # TODO:
        return label_info

    def generate_patch_fn(self, case_info, patch_loc, label_info=None):
        if label_info is None:
            tmp = (case_info["fn_str"] + "_%d_%d_" + self.save_format) % (int(patch_loc[0]), int(patch_loc[1]))
            fn = os.path.join(self.save_dir, tmp)
        else:
            fn = "temp"+self.save_format
            # TODO:
            print("TODO: add label to file name")
        return fn

    def generate_tfRecord_fp(self, case_info, feature_map):
        tmp = case_info["fn_str"] + self.save_format
        fn = os.path.join(self.save_dir, tmp)
        writer = tf.python_io.TFRecordWriter(fn)  # generate tfRecord file handle
        return writer, fn

    # get image patches and write to files
    def save_patch_without_annotation(self, wsi_obj, case_info, indices):
        patch_cnt = 0
        if self.with_feature_map:
            tf_writer, tf_fn = self.generate_tfRecord_fp(case_info, self.feature_map)
        [loc_x, loc_y] = indices
        for idx, lx in enumerate(loc_x):
            # if logging.DEBUG == logging.root.level:
            #     import matplotlib.pyplot as plt
            #     plt.figure(1)
            #     plt.plot(loc_x, loc_y, 'r.')
            #     plt.plot(loc_x[idx], loc_y[idx], 'go')
            #     plt.gca().invert_yaxis()
            #     plt.show()
            patch = wsi_obj.read_region((loc_x[idx], loc_y[idx]), self.extract_layer, (self.patch_size, self.patch_size)).convert("RGB")
            Content_rich = True
            if self.patch_filter_by_area:  # if we need to filter the image patch
                Content_rich = self.filter_by_content_area(np.array(patch), area_threshold=0.5)
            if Content_rich:
                patch_cnt += 1
                if self.with_feature_map:  # Append data to tfRecord file
                    # TODO: write the data into the customized key-value map, maybe need to implement in another class
                    values = []
                    for eval_str in self.feature_map.eval_str:
                        values.append(eval(eval_str))
                    features = self.feature_map.update_feature_map_eval(values)
                    example = tf.train.Example(features=tf.train.Features(feature=features))  # Create an example protocol buffer
                    tf_writer.write(example.SerializeToString())  # Serialize to string and write on the file
                else:  # save patch to jpg, with label text and id in file name
                    if logging.DEBUG == logging.root.level:
                        import matplotlib.pyplot as plt
                        plt.figure(1)
                        plt.imshow(patch)
                        plt.show()
                    fn = self.generate_patch_fn(case_info, (loc_x[idx], loc_y[idx]))
                    if self.save_format == ".jpg":
                        if patch.mode == "RGBA":
                            patch = patch.convert("RGB")
                        patch.save(fn)
                    elif self.save_format == ".png":
                        if patch.mode == "RGB":
                            patch = patch.convert("RGBA")
                        patch.save(fn)
                    else:
                        raise Exception("Can't recognize save format")
            else:
                logging.debug("Ignore the patch")
        tf_writer.close()
        return patch_cnt

    def extract(self, wsi_fn):
        wsi_obj, case_info = self.get_case_info(wsi_fn)
        wsi_thumb = self.get_thumbnail(wsi_obj)    # get the thumbnail
        wsi_thumb_mask = self.tissue_detector.predict(wsi_thumb)   # get the foreground thumbnail mask
        if logging.DEBUG == logging.root.level:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(2, 1)
            ax[0].imshow(wsi_thumb)
            ax[1].imshow(wsi_thumb_mask, cmap='gray')
            plt.show()
        if not self.with_anno:
            return self.save_patch_without_annotation(wsi_obj, case_info, self.get_patch_locations(wsi_thumb_mask))
        else:
            print("TODO: extract patches with annotations")


# example
if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.DEBUG)
    # logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.ERROR)

    # example of single case
    wsi_fn = "/projects/shart/digital_pathology/data/PenMarking/WSIs/MELF-Clean/8a26a55a78b947059da4e8c36709a828.tiff"
    # wsi_fn = "/projects/shart/digital_pathology/data/PenMarking/WSIs/MELF/d83cc7d1c941438e93786fc381ab5bb5.tiff"
    # wsi_fn = "/projects/shart/digital_pathology/data/PenMarking/WSIs/MELF/7bb50b5d9dcf4e53ad311d66136ae00f.tiff"
    gnb_training_files = "/projects/shart/digital_pathology/data/PenMarking/model/tissue_loc/HE_tissue_others.tsv"

    from wsitools.tissue_detection.tissue_detector import TissueDetector  # import dependent packages
    from wsitools.patch_extraction.feature_map_creator import FeatureMapCreator

    tissue_detector = TissueDetector("GNB", threshold=0.5, training_files=gnb_training_files)

    # extract patches without annotation, no feature map specified and save patches to '.jpg'
    output_dir = "/projects/shart/digital_pathology/data/PenMarking/temp"

    # # Save to JPG/PNG files
    # parameters = ExtractorParameters(output_dir, save_format='.png', sample_cnt=-1)
    # patch_extractor = PatchExtractor(tissue_detector, parameters, feature_map=None, annotations=None)
    # patch_num = patch_extractor.extract(wsi_fn)

    # # Save to tfRecords
    fm = FeatureMapCreator("./feature_maps/basic_fm_P_eval.csv")
    parameters = ExtractorParameters(output_dir, save_format='.tfRecord', sample_cnt=-1)
    patch_extractor = PatchExtractor(tissue_detector, parameters, feature_map=fm, annotations=None)
    patch_num = patch_extractor.extract(wsi_fn)
    print("%d Patches have been save to %s" % (patch_num, output_dir))

    # #example of multiple cases
    # Save to JPG/PNG files (passed)
    # from wsitools.file_management.case_list_manager import CaseListManager
    # import multiprocessing
    #
    # case_list_txt = "/projects/shart/digital_pathology/data/PenMarking/WSIs/annotated_cases.txt"
    # case_mn = CaseListManager(case_list_txt)
    # all_wsi_fn = case_mn.case_list
    # parameters = ExtractorParameters(output_dir, save_format='.png', sample_cnt=-1)
    # patch_extractor = PatchExtractor(tissue_detector, parameters, feature_map=None, annotations=None)
    #
    # multiprocessing.set_start_method('spawn')
    # pool = multiprocessing.Pool(processes=4)
    # pool.map(patch_extractor.extract, all_wsi_fn)







