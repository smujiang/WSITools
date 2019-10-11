# Patch Extraction
This module aims at extracting patches from a pair of whole slide images.
  
Like [extract patches from a single case](./patch_extraction.md), you also need to [define a tissue detector](../tissue_detection/tissue_detector.md) to identify the foreground of a WSI, because we don't want to wast our time on blank patches. Other than that you may also need a ```WSI_CaseManager``` to help you to find the WSI counterpart, which maintains the correspondence in a MS Excel file.  
In order to create the patch location correspondence, we also need a ```OffsetCSVManager``` to help us to maintain the shifting offsets between WSI pairs. The offsets can be obtained from both [automatic registration](../wsi_registration/auto_registration.md) and [annotation](../wsi_annotation/QuPath_scripts/readme.md). 

You may also need to specify some parameters to customise your extraction. All the parameters are warped in ```PairwiseExtractorParameters``` of [pairwise_patch_extractor.py](../../src/patch_extraction/pairwise_patch_extractor.py). Read the comments in the file to get more details. 

### Extract patches from a single pair of WSIs
To extract patches from a single WSI, you can write your code like below.    
Currently, we don't provide acceleration for single case patch extraction
```python

```

### Extract patches from multiple pairs of WSIs
Multiprocessing can be adopted to accelerate the extraction.
```python

```
























