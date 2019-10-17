import sys
if sys.version_info[0] < 3:
    raise Exception("Error: You are not running Python 3.")

from file_management import case_list_manager, offset_csv_manager, wsi_case_manager, class_label_csv_manager
from tissue_detection import tissue_detector
from wsi_annotation import offset_annotation, region_annotation
from wsi_registration import auto_wsi_matcher