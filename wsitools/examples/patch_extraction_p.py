from wsitools.tissue_detection.tissue_detector import TissueDetector
from wsitools.patch_extraction.patch_extractor import ExtractorParameters, PatchExtractor

wsi_fn = "/data/WSIs/MELF-Clean/8a26a55a78b947059da4e8c36709a828.tiff"
output_dir = "/data/WSIs_extraction"

tissue_detector = TissueDetector("LAB_Threshold", threshold=85)
parameters = ExtractorParameters(output_dir, save_format='.png', sample_cnt=-1)
patch_extractor = PatchExtractor(tissue_detector, parameters, feature_map=None, annotations=None)
patch_num = patch_extractor.extract(wsi_fn)