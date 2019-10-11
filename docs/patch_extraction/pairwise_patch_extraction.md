# Pairwise Patch Extraction
This module aims at extracting patches from a pair of whole slide images.
  
To start patch extraction, like [extract patches from a single case](./patch_extraction.md), you also need to [define a tissue detector](../tissue_detection/tissue_detector.md) to identify the foreground of a WSI. Other than that you may also need a ```WSI_CaseManager``` to help you to find the WSI counterpart, which maintains the correspondence in a MS Excel file.  
In order to create the patch location correspondence, we also need a ```OffsetCSVManager``` to help us to maintain the shifting offsets between WSI pairs. The offsets can be obtained from both [automatic registration](../wsi_registration/wsi_registration.md) and [annotation](../wsi_annotation/QuPath_scripts/readme.md). 

You may also need to specify some parameters to customise your extraction. All the parameters are warped in ```PairwiseExtractorParameters``` of [pairwise_patch_extractor.py](../../wsitools/patch_extraction/pairwise_patch_extractor.py). Read the comments in the file to get more details. 

### Extract patches from a single pair of WSIs
To extract patches from a single WSI, you can write your code like below.    
Currently, we don't provide acceleration for single case patch extraction
```python
fixed_wsi = "/projects/WSIs/d83cc7d1c941438e93786fc381ab5bb5.tiff"
float_wsi_root_dir = "/projects/WSIs/MELF-Clean"
gnb_training_files = "../tissue_detection/model_files/tissue_others.tsv"

from WSItools.file_managment.wsi_case_manager import WSI_CaseManager  # import dependent packages
from WSItools.file_managment.offset_csv_manager import OffsetCSVManager
from WSItools.tissue_detection.tissue_detector import TissueDetector
from WSItools.patch_extraction.pairwise_patch_extractor import PairwiseExtractorParameters, PairwisePatchExtractor

tissue_detector = TissueDetector("GNB", threshold=0.5, training_files=gnb_training_files)

case_mn = WSI_CaseManager()
float_wsi = case_mn.get_counterpart_fn(fixed_wsi, float_wsi_root_dir)
_, fixed_wsi_uuid, _ = case_mn.get_wsi_fn_info(fixed_wsi)
_, float_wsi_uuid, _ = case_mn.get_wsi_fn_info(float_wsi)
offset_csv_fn = "/projects/WSIs_aux/registration_offsets.csv"
offset_csv_mn = OffsetCSVManager(offset_csv_fn)
offset, state_indicator = offset_csv_mn.lookup_table(fixed_wsi_uuid, float_wsi_uuid)
if state_indicator == 0:
    raise Exception("No corresponding offset can be found in the file")

# extract pairs of patches without annotation, no feature map specified and save patches to '.jpg'
output_dir = "/projects/data/temp"
parameters = PairwiseExtractorParameters(output_dir, save_format='.jpg', sample_cnt=-1)
patch_extractor = PairwisePatchExtractor(tissue_detector, parameters, feature_map=None, annotations=None)
patch_cnt = patch_extractor.extract(fixed_wsi, float_wsi, offset)

print("%d Patches have been save to %s" % (patch_cnt, output_dir))
```

### Extract patches from multiple pairs of WSIs
Multiprocessing can be adopted to accelerate the extraction.
```python

```
























