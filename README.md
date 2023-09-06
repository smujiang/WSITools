# WSITools
Tools for whole slide image (WSI) pre-processing, including tissue detection, patch extraction, annotation parsing etc.
# Citation
Use this bibtex to cite this repository:
```
@misc{Jun Jiang WSITools 2019,
  title={Whole slide image pre-processing tools for deep learning tasks},
  author={Jun Jiang},
  year={2019},
  publisher={Github},
  journal={GitHub repository},
  howpublished={\url{https://github.com/smujiang/WSITools}},
}
```
Or cite our paper used this tool
```
[1] Jiang, Jun, Burak Tekin, Lin Yuan, Sebastian Armasu, Stacey Winham, E. Goode, Hongfang Liu, Yajue Huang, Ruifeng Guo, and Chen Wang. "Computational tumor stroma reaction evaluation led to novel prognosis-associated fibrosis and molecular signature discoveries in high-grade serous ovarian carcinoma." Frontiers in medicine 9 (2022).
[2] Jiang, Jun, Burak Tekin, Ruifeng Guo, Hongfang Liu, Yajue Huang, and Chen Wang. "Digital pathology-based study of cell-and tissue-level morphologic features in serous borderline ovarian tumor and high-grade serous ovarian cancer." Journal of Pathology Informatics 12 (2021).
[3] Jiang, Jun, Naresh Prodduturi, David Chen, Qiangqiang Gu, Thomas Flotte, Qianjin Feng, and Steven Hart. "Image-to-image translation for automatic ink removal in whole slide images." Journal of Medical Imaging 7, no. 5 (2020): 057502.
```
## Quick Start
### Installation
```bash
git clone https://github.com/smujiang/WSITools.git
cd WSITools
python setup.py install
```
* Note that when you install our package, the dependencies can be automatically installed, but you may need to install 
the dependent [OpenSlide](https://openslide.org/) library.
  * If using `PyCharm` or `venv` on Windows:
    1. Download the correct [binary](https://openslide.org/download/#windows-binaries) file for your system
    2. Copy all files from `/bin` into your `venv/Scripts/` directory

### Testing
We provide examples for [Patch Extraction](docs/patch_extraction/patch_extraction.md) and 
[Pairwise Patch Extraction](docs/patch_extraction/pairwise_patch_extraction.md). You can choose to save the extracted 
patches into PNG/JPG files or [tfRecords](https://www.tensorflow.org/tutorials/load_data/tfrecord).

If you just want to extract patches from a WSI, and save them into JPG/PNG files, it needs only a few lines of code:
```python
# Import the relevant libraries from this module
from wsitools.tissue_detection.tissue_detector import TissueDetector
from wsitools.patch_extraction.patch_extractor import ExtractorParameters, PatchExtractor
import multiprocessing

#Define some run parameters
num_processors = 20                     # Number of processes that can be running at once
wsi_fn = "/path/2/file.tff"             # Define a sample image that can be read by OpenSlide
output_dir = "/data/WSIs_extraction"    # Define an output directory

# Define the parameters for Patch Extraction, including generating an thumbnail from which to traverse over to find 
# tissue.
parameters = ExtractorParameters(output_dir, # Where the patches should be extracted to
    save_format = '.png',                      # Can be '.jpg', '.png', or '.tfrecord'              
    sample_cnt = -1,                           # Limit the number of patches to extract (-1 == all patches)
    patch_size = 128,                          # Size of patches to extract (Height & Width)
    rescale_rate = 128,                        # Fold size to scale the thumbnail to (for faster processing)
    patch_filter_by_area = 0.5,                # Amount of tissue that should be present in a patch
    with_anno = True,                          # If true, you need to supply an additional XML file
    extract_layer = 0                          # OpenSlide Level
    )

# Choose a method for detecting tissue in thumbnail image
tissue_detector = TissueDetector("LAB_Threshold",   # Can be LAB_Threshold or GNB
    threshold = 85,                                   # Number from 1-255, anything less than this number means there is tissue
    training_files = None                             # Training file for GNB-based detection
    )

# Create the extractor object
patch_extractor = PatchExtractor(tissue_detector, 
    parameters, 
    feature_map = None,                       # See note below                     
    annotations = None                        # Object of Annotation Class (see other note below)
    )

if __name__ == '__main__':
    # Run the extraction process
    multiprocessing.set_start_method('spawn')
    pool = multiprocessing.Pool(processes = num_processors)
    pool.map(patch_extractor.extract, [wsi_fn])

```
> See [Feature Maps](docs/patch_extraction/feature_map.md) for more detail

> See [Annotation Objects](docs/patch_extraction/annotation.md) for more detail

## Descriptions
WSITools is a whole slide image processing toolkit. It provides efficient ways to extract patches from whole slide 
images, and some other useful features for pathological image processing.
Currently, it supports four patch extraction scenarios:
1. Extract patches from WSIs
2. Extract patches from WSIs and their label (i.e. their directory name)
    1. TODO: Incomplete
3. Extract patches from a fixed and a float WSI
4. Extract patches from a fixed and a float WSI in places that intersect annotation objects
    1. TODO: Incomplete

## Additional Features
1. Detect tissue in a WSI
2. Export and parsing annotation from [QuPath](https://qupath.github.io/) and [Aperio Image Scope](https://www.leicabiosystems.com/digital-pathology/manage/aperio-imagescope/) 
3. WSI registration for image pairs [[Paper]](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0220074)
4. Reconstruct WSI from the processed image patches

## Architectures
![Architecture](docs/imgs/arch.png)
## Documents
[Tissue Detection](docs/tissue_detection/tissue_detector.md)   
[Patch Extraction](docs/patch_extraction/patch_extraction.md)   
[WSI Alignment](docs/wsi_registration/wsi_registration.md)          
[Pairwise Patch Extraction](docs/patch_extraction/pairwise_patch_extraction.md)   
[Annotate with QuPath and Export Annotations](docs/wsi_annotation/QuPath_scripts/readme.md)  
[Annotation Parsing](docs/wsi_annotation/annotation_parsing.md)
## TODO list
* Validate saved tfRecords.
* Add annotation labels into patch extraction.
