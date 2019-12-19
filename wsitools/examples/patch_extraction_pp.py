fixed_wsi = "/projects/shart/digital_pathology/data/PenMarking/WSIs/MELF/e39a8d60a56844d695e9579bce8f0335.tiff"
float_wsi_root_dir = "/projects/shart/digital_pathology/data/PenMarking/WSIs/MELF-Clean/"

from wsitools.file_management.wsi_case_manager import WSI_CaseManager  # import dependent packages
from wsitools.file_management.offset_csv_manager import OffsetCSVManager
from wsitools.tissue_detection.tissue_detector import TissueDetector
from wsitools.patch_extraction.pairwise_patch_extractor import PairwiseExtractorParameters, PairwisePatchExtractor

gnb_training_files = "../tissue_detection/model_files/HE_tissue_others.tsv"
tissue_detector = TissueDetector("GNB", threshold=0.5, training_files=gnb_training_files)

case_mn = WSI_CaseManager()
float_wsi = case_mn.get_counterpart_fn(fixed_wsi, float_wsi_root_dir)
_, fixed_wsi_uuid, _ = case_mn.get_wsi_fn_info(fixed_wsi)
_, float_wsi_uuid, _ = case_mn.get_wsi_fn_info(float_wsi)
offset_csv_fn = "../file_management/example/wsi_pair_offset.csv"
offset_csv_mn = OffsetCSVManager(offset_csv_fn)
offset, state_indicator = offset_csv_mn.lookup_table(fixed_wsi_uuid, float_wsi_uuid)
if state_indicator == 0:
    raise Exception("No corresponding offset can be found in the file")
output_dir = "/projects/shart/digital_pathology/data/PenMarking/temp"
parameters = PairwiseExtractorParameters(output_dir, save_format='.jpg', sample_cnt=-1)
patch_extractor = PairwisePatchExtractor(tissue_detector, parameters, feature_map=None, annotations=None)
patch_cnt = patch_extractor.extract(fixed_wsi, float_wsi, offset)
print("%d Patches have been save to %s" % (patch_cnt, output_dir))
