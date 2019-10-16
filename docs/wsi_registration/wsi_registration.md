# WSI registration (Alignment)
Occasionally, we need to scan the slide twice (scan the slide and then treat the slide with other chemicals and scan it again). In this scenario, we need to align the two WSI to establish the tissue correspondence of two scans.  

Ideally, this is a rigid image registration problem, just shifting, even without rotation (According to the way of mounting the slides, their should be no rotation of two scans.)
So in current version, we don't take the rotation into account, because the rotation angle is so tiny and can be ignored. 

To standardize the offset saving and loading, we introduced a [csv file](../../wsitools/file_management/example/wsi_pair_offset.csv) to maintain the WSI pairs and their offsets.

We provide two ways to get the shifting offset: 
### 1. Automatic registration
Here is the an example of how to automatically align two WSIs and save/load the offset to/from the csv file. It's a easy-to-use version of [our previous work](https://github.com/smujiang/Re-stained_WSIs_Registration). 

* Note that rotation is not taken into account in this automatic registration. 
* In most WSI scanner, the way of mounting slides determined that there should be minimum rotation, practically almost zero. 
* So if the slide is scanned and re-scanned, the obtained two WSIs should be no rotation.
* But if the WSIs are not obtained in scan and re-scan way, the rotation shouldn't be ignored, and you can not use this automatic registration.

You need to [define a tissue detector](../tissue_detection/tissue_detector.md) to identify the foreground of a WSI, from which image patches will be extracted, and shifting offset will be calculated based on these patches.
Additionally, you may also need a ```WSI_CaseManager``` to help you to find the WSI counterpart, which maintains the correspondence in a MS Excel file. 
```python
from wsitools.file_management.wsi_case_manager import WSI_CaseManager   # import dependent packages
from wsitools.file_management.offset_csv_manager import OffsetCSVManager
from wsitools.tissue_detection.tissue_detector import TissueDetector
from wsitools.wsi_registration.auto_wsi_matcher import MatcherParameters, WSI_Matcher
fixed_wsi = "/projects/MELF/7bb50b5d9dcf4e53ad311d66136ae00f.tiff"
#float_wsi = "/projects/MELF-Clean/8a26a55a78b947059da4e8c36709a828.tiff"
float_wsi_root_dir = "/projects/MELF-Clean/"


case_mn = WSI_CaseManager()
float_wsi = case_mn.get_counterpart_fn(fixed_wsi, float_wsi_root_dir)
_, fixed_wsi_uuid, _ = case_mn.get_wsi_fn_info(fixed_wsi)
_, float_wsi_uuid, _ = case_mn.get_wsi_fn_info(float_wsi)

offset_csv_fn = "../file_management/example/registration_offsets.csv"
offset_csv_mn = OffsetCSVManager(offset_csv_fn)
offset_tmp, state_indicator = offset_csv_mn.lookup_table(fixed_wsi_uuid, float_wsi_uuid)
if state_indicator == 0 or staticmethod == 1:     # Auto registration does not exist
    tissue_detector = TissueDetector("GNB", threshold=0.5)
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
``` 
### 2. QuPath annotation  
We provide this [Annotate with QuPath and Export Annotations](../wsi_annotation/QuPath_scripts/readme.md)  document shows how to annotate with QuPath, export the annotation and calculate offset with python.









