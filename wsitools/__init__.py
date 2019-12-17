import sys
if sys.version_info[0] < 3:
    raise Exception("Error: You are not running Python 3.")

from wsitools.file_management import case_list_manager, offset_csv_manager, wsi_case_manager, class_label_csv_manager
from wsitools.tissue_detection import tissue_detector
from wsitools.wsi_annotation import point_annotation, region_annotation
from wsitools.wsi_registration import auto_wsi_matcher
