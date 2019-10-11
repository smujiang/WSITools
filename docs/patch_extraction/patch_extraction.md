# Patch Extraction
This module aims at extracting patches from whole slide images, and save the export patches into jpg/png/tfRecords files, depends on your parameter settings.   

To start patch extraction, first, you need to [define a tissue detector](../tissue_detection/tissue_detector.md) to identify the foreground of a WSI, because we don't want to wast our time on blank patches. Then, you need to specify some parameters to customise your extraction, anyhow you may use the default parameters. All the parameters are warped in ```ExtractorParameters``` of [patch_extractor.py](../../wsitools/patch_extraction/patch_extractor.py). Read the comments in the file to get more details. 

### Extract patches from a single WSI
To extract patches from a single WSI, you can write your code like below.    
Currently, we don't provide acceleration for single case patch extraction
```python
from wsitools.patch_extraction.patch_extractor import ExtractorParameters, PatchExtractor

wsi_fn = "/data/8a26a55a78b947059da4e8c36709a828.tiff" # WSI file name
gnb_training_files = "./model_files/tissue_others.tsv"

from wsitools.tissue_detection.tissue_detector import TissueDetector  # import dependent packages
tissue_detector = TissueDetector("GNB", threshold=0.5, training_files=gnb_training_files)

# extract patches without annotation, no feature map specified and save patches to '.jpg'
output_dir = "/data/wsi_patches"
parameters = ExtractorParameters(output_dir, save_format='.jpg', sample_cnt=-1)
patch_extractor = PatchExtractor(tissue_detector, parameters, feature_map=None, annotations=None)
patch_num = patch_extractor.extract(wsi_fn)
print("%d Patches have been save to %s" % (patch_num, output_dir))
```

### Extract patches from a list of WSI
Multiprocessing can be adopted to accelerate the extraction.
```python

```















