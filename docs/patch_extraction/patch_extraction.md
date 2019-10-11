#Patch Extraction

### Extract patches from a single WSI
To extract patches from a single WSI, you can write your code like below.    
Be aware, you need to [define a tissue detector](../tissue_detection/tissue_detector.md) to identify the foreground of a WSI, because we don't want to wast our time on blank patches.

```python
from WSItools.patch_extraction import ExtractorParameters, PatchExtractor

wsi_fn = "/data/8a26a55a78b947059da4e8c36709a828.tiff" # WSI file name
gnb_training_files = "./model_files/tissue_others.tsv"

from src.tissue_detection.tissue_detector import TissueDetector  # import dependent packages
tissue_detector = TissueDetector("GNB", threshold=0.5, training_files=gnb_training_files)

# extract patches without annotation, no feature map specified and save patches to '.jpg'
output_dir = "/data/wsi_patches"
parameters = ExtractorParameters(output_dir, save_format='.jpg', sample_cnt=-1)
patch_extractor = PatchExtractor(tissue_detector, parameters, feature_map=None, annotations=None)
patch_num = patch_extractor.extract(wsi_fn)
print("%d Patches have been save to %s" % (patch_num, output_dir))

```














