# Patch Extraction
This module aims at extracting patches from whole slide images, and save the export patches into jpg/png/tfRecords files, depends on your parameter settings.   

To start patch extraction, first, you need to [define a tissue detector](../tissue_detection/tissue_detector.md) to identify the foreground of a WSI, because we don't want to wast our time on blank patches. Then, you need to specify some parameters to customise your extraction, anyhow you may use the default parameters. All the parameters are warped in ```ExtractorParameters``` of [patch_extractor.py](../../wsitools/patch_extraction/patch_extractor.py). Read the comments in the file to get more details. 

## Extract patches from a single WSI, save to JPEG files
To extract patches from a single WSI, you can write your code like below.    
Currently, we don't provide acceleration for single case patch extraction
```python
from wsitools.patch_extraction.patch_extractor import ExtractorParameters, PatchExtractor
from wsitools.tissue_detection.tissue_detector import TissueDetector  # import dependent packages

wsi_fn = "/data/8a26a55a78b947059da4e8c36709a828.tiff" # WSI file name

gnb_training_files = "../tissue_detection/model_files/HE_tissue_others.tsv"
tissue_detector = TissueDetector("GNB", threshold=0.5, training_files=gnb_training_files)

# extract patches without annotation, no feature map specified and save patches to '.jpg'
output_dir = "/data/wsi_patches"
parameters = ExtractorParameters(output_dir, save_format='.jpg', sample_cnt=-1)
patch_extractor = PatchExtractor(tissue_detector, parameters, feature_map=None, annotations=None)
patch_num = patch_extractor.extract(wsi_fn)
print("%d Patches have been save to %s" % (patch_num, output_dir))
```

## Extract patches from a list of WSIs, save to PNG files
All the file names of WSIs to be processed can be listed in a text file [(example)](../../wsitools/file_management/example/case_list.txt), so that it can be easily managed.
Multiprocessing can be adopted to accelerate the extraction.
```python
from wsitools.tissue_detection.tissue_detector import TissueDetector 
from wsitools.patch_extraction.patch_extractor import ExtractorParameters, PatchExtractor
from wsitools.file_management.case_list_manager import CaseListManager
import multiprocessing

case_list_txt = "../file_management/example/case_list.txt"
case_mn = CaseListManager(case_list_txt)
all_wsi_fn = case_mn.case_list

gnb_training_files = "../tissue_detection/model_files/HE_tissue_others.tsv"
tissue_detector = TissueDetector("GNB", threshold=0.5, training_files=gnb_training_files)

output_dir = "/data/wsi_patches"
parameters = ExtractorParameters(output_dir, save_format='.png', sample_cnt=-1)
patch_extractor = PatchExtractor(tissue_detector, parameters, feature_map=None, annotations=None)

multiprocessing.set_start_method('spawn')
pool = multiprocessing.Pool(processes=4)
pool.map(patch_extractor.extract, all_wsi_fn)
```
## Extract patches from a single WSI, save to tfRecords
TensorFlow tfRecord provide an efficient way to write and read structured data.
If you would like to save the extracted patches and some other information into tfRecords, you may need to [customize your own feature map](./feature_map.md).

```python
from wsitools.patch_extraction.patch_extractor import ExtractorParameters, PatchExtractor
from wsitools.tissue_detection.tissue_detector import TissueDetector  # import dependent packages
from wsitools.patch_extraction.feature_map_creator import FeatureMapCreator

wsi_fn = "/data/8a26a55a78b947059da4e8c36709a828.tiff" # WSI file name

gnb_training_files = "../tissue_detection/model_files/HE_tissue_others.tsv"
tissue_detector = TissueDetector("GNB", threshold=0.5, training_files=gnb_training_files)

output_dir = "/data/wsi_patches"
fm = FeatureMapCreator("./feature_maps/basic_fm_P_eval.csv")
parameters = ExtractorParameters(output_dir, save_format='.tfRecord', sample_cnt=-1)
patch_extractor = PatchExtractor(tissue_detector, parameters, feature_map=fm, annotations=None)
patch_num = patch_extractor.extract(wsi_fn)
print("%d Patches have been save to %s" % (patch_num, output_dir))
```












