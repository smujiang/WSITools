from wsitools.tissue_detection.tissue_detector import TissueDetector
from wsitools.patch_extraction.patch_extractor import ExtractorParameters, PatchExtractor

wsi_fn = "/infodev1/non-phi-data/junjiang/OvaryCancer/WSIs/OCMC-016.svs"  # WSI file name
output_dir = "/infodev1/non-phi-data/junjiang/OvaryCancer/Patches/h5_files"
log_dir = "/infodev1/non-phi-data/junjiang/OvaryCancer/Patches/logs"


tissue_detector = TissueDetector("LAB_Threshold", threshold=85)  #

parameters = ExtractorParameters(output_dir, log_dir=log_dir, patch_size=500, stride=500, extract_layer=0, save_format='.h5', sample_cnt=-1)

patch_extractor = PatchExtractor(tissue_detector, parameters=parameters)
patch_num = patch_extractor.extract(wsi_fn)


print("%d Patches have been save to %s" % (patch_num, output_dir))


