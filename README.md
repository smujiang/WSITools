# WSITools
Tools for whole slide image (WSI) pre-processing
## Quick Start
### Installation
```bash
git clone https://github.com/smujiang/WSITools.git
cd WSITools
python setup.py install
```
* Note that the dependencies can automatically installed, but you may need to install the dependent [OpenSlide](https://openslide.org/) library.
### Testing
We provide examples for [Patch Extraction](docs/patch_extraction/patch_extraction.md) and [Pairwise Patch Extraction](docs/patch_extraction/pairwise_patch_extraction.md). Multiple processing is available for multiple WSIs extractions.

## Descriptions
WSITools is a whole slide image pre-processing kit. It provide efficient way to extract patches from whole slide images.
Current, it supports four patch extraction scenarios:
1. extract patches from WSIs (P)
2. extract patches and their labels from WSIs and their annotations (P+L)
3. extract patches and their labels from WSIs as well as their counterparts (P+P')
4. extract patches and their labels from WSIs as well as their counterparts with their annotations (P+P'+L)

For now,
* We can only save the extracted patches into PNG/JPG files. We are working on saving them into tfRecords.
* We are working on adding annotation labels into patch extraction.

## Architectures
![Architecture](docs/imgs/arch.png)
## Documents
[Tissue Detection](docs/tissue_detection/tissue_detector.md)   
[Patch Extraction](docs/patch_extraction/patch_extraction.md)
[WSI Alignment](docs/wsi_registration/wsi_registration.md)       
[Pairwise Patch Extraction](docs/patch_extraction/pairwise_patch_extraction.md)
[Annotate with QuPath and Export Annotations](docs/wsi_annotation/QuPath_scripts/readme.md)

