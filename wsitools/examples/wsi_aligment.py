from wsitools.file_management.wsi_case_manager import WSI_CaseManager
from wsitools.file_management.offset_csv_manager import OffsetCSVManager
from wsitools.tissue_detection.tissue_detector import TissueDetector
from wsitools.wsi_registration.auto_wsi_matcher import MatcherParameters, WSI_Matcher

# fixed_wsi = "/projects/shart/digital_pathology/data/PenMarking/WSIs/MELF/7bb50b5d9dcf4e53ad311d66136ae00f.tiff"
fixed_wsi = "/projects/shart/digital_pathology/data/PenMarking/WSIs/MELF/e39a8d60a56844d695e9579bce8f0335.tiff"
#float_wsi = "/projects/shart/digital_pathology/data/PenMarking/WSIs/MELF-Clean/8a26a55a78b947059da4e8c36709a828.tiff"
float_wsi_root_dir = "/projects/shart/digital_pathology/data/PenMarking/WSIs/MELF-Clean/"


case_mn = WSI_CaseManager()
float_wsi = case_mn.get_counterpart_fn(fixed_wsi, float_wsi_root_dir)
_, fixed_wsi_uuid, _ = case_mn.get_wsi_fn_info(fixed_wsi)
_, float_wsi_uuid, _ = case_mn.get_wsi_fn_info(float_wsi)

offset_csv_fn = "/projects/shart/digital_pathology/data/PenMarking/WSIs/registration_offsets.csv"
offset_csv_mn = OffsetCSVManager(offset_csv_fn)
offset_tmp, state_indicator = offset_csv_mn.lookup_table(fixed_wsi_uuid, float_wsi_uuid)
if state_indicator == 0 or state_indicator == 1:     # Auto registration does not exist
    gnb_training_tsv = "../tissue_detection/model_files/HE_tissue_others.tsv"
    tissue_detector = TissueDetector("GNB", threshold=0.5, training_files=gnb_training_tsv)
    matcher_parameters = MatcherParameters()
    matcher = WSI_Matcher(tissue_detector, matcher_parameters)
    offset = matcher.match(fixed_wsi, float_wsi)
    offset_csv_mn.update_auto_registration(fixed_wsi_uuid, float_wsi_uuid, offset)
    print("Automatic registration does not exist in file: %s." % offset_csv_fn)
    print("Add automatic registration result: (%.2f %.2f) to this file" % (offset[0], offset[1]))
else:
    print("Automatic registration is already in the csv file: %s" % offset_csv_fn)
    print("Quarried registration result: (%.2f %.2f)" % (offset_tmp[0], offset_tmp[1]))  # could be ground truth

if state_indicator == 0 or state_indicator == 2:
    print("Ground truth does not exist in file: %s. Need to update ground truth" % offset_csv_fn)
    print("Get ground truth from 'Annotation' and call OffsetCSVManager.update_ground_truth(...)")
else:
    print("Looking up ground truth offset from %s" % offset_csv_fn)
    print("Found ground truth offset: (%.2f %.2f)" % (offset_tmp[0], offset_tmp[1]))