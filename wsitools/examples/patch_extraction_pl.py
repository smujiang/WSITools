from wsitools.tissue_detection.tissue_detector import TissueDetector
from wsitools.patch_extraction.feature_map_creator import FeatureMapCreator
from wsitools.wsi_annotation.region_annotation import AnnotationRegions
from wsitools.patch_extraction.patch_extractor import ExtractorParameters, PatchExtractor

wsi_fn = "/projects/shart/digital_pathology/data/PenMarking/WSIs/MELF/e39a8d60a56844d695e9579bce8f0335.tiff"  # WSI file name
output_dir = "/projects/shart/digital_pathology/data/PenMarking/temp"

tissue_detector = TissueDetector("LAB_Threshold", threshold=85)  #
fm = FeatureMapCreator("../patch_extraction/feature_maps/basic_fm_PL_eval.csv")  # use this template to create feature map
xml_fn = "../wsi_annotation/examples/e39a8d60a56844d695e9579bce8f0335.xml"
class_label_id_csv = "../wsi_annotation/examples/class_label_id.csv"
annotations = AnnotationRegions(xml_fn, class_label_id_csv)

parameters = ExtractorParameters(output_dir, save_format='.tfrecord', sample_cnt=-1)
patch_extractor = PatchExtractor(tissue_detector, parameters, feature_map=fm, annotations=annotations)
patch_num = patch_extractor.extract(wsi_fn)
print("%d Patches have been save to %s" % (patch_num, output_dir))
