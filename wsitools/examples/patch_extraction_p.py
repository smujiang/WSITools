from wsitools.tissue_detection.tissue_detector import TissueDetector
from wsitools.patch_extraction.patch_extractor import ExtractorParameters, PatchExtractor
import multiprocessing
import os

wsi_fn_list = [os.path.join("/infodev1/non-phi-data/junjiang/OvaryCancer/WSIs", "OCMC-{:03d}.svs".format(i)) for i in range(1, 31)]


output_dir = "/infodev1/non-phi-data/junjiang/OvaryCancer/Patches/h5_files"

log_dir = "/infodev1/non-phi-data/junjiang/OvaryCancer/Patches/logs"


# Define some run parameters
num_processors = 10  # Number of processes that can be running at once

tissue_detector = TissueDetector("LAB_Threshold", threshold=85)  #

parameters = ExtractorParameters(output_dir, log_dir=log_dir, patch_size=500, stride=500, extract_layer=0, save_format='.h5',  sample_cnt=-1)

patch_extractor = PatchExtractor(tissue_detector, parameters=parameters)


pool = multiprocessing.Pool(processes=num_processors)
pool.map(patch_extractor.extract, wsi_fn_list)




