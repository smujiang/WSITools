from wsitools.tissue_detection.tissue_detector import TissueDetector
from wsitools.patch_extraction.feature_map_creator import FeatureMapCreator
from wsitools.patch_extraction.patch_extractor import ExtractorParameters, PatchExtractor
import multiprocessing

#Define some run parameters
num_processors = 20                     # Number of processes that can be running at once
wsi_fn = "/infodev1/non-phi-data/junjiang/OvaryCancer/WSIs/1084181_CR02-2502-A2_HE.svs"           # Define a sample image that can be read by OpenSlide
output_dir = "/infodev1/non-phi-data/junjiang/OvaryCancer/Patches_out"    # Define an output directory
log_dir = "/infodev1/non-phi-data/junjiang/OvaryCancer/Patches_out"

fm = FeatureMapCreator("../patch_extraction/feature_maps/basic_fm_P_eval.csv")  # use this template to create feature map


# Define the parameters for Patch Extraction, including generating an thumbnail from which to traverse over to find
# tissue.
parameters = ExtractorParameters(output_dir, # Where the patches should be extracted to
    save_format='.tfrecord',                      # Can be '.jpg', '.png', or '.tfrecord'
    sample_cnt=200,                           # Limit the number of patches to extract (-1 == all patches)
    patch_size=512,                          # Size of patches to extract (Height & Width)
    stride=512,
    rescale_rate=128,                        # Fold size to scale the thumbnail to (for faster processing)
    patch_filter_by_area=0.5,                # Amount of tissue that should be present in a patch
    with_anno=True,                          # If true, you need to supply an additional XML file
    log_dir=log_dir,
    extract_layer=0                          # OpenSlide Level

    )

# Choose a method for detecting tissue in thumbnail image
tissue_detector = TissueDetector("LAB_Threshold",   # Can be LAB_Threshold or GNB
    threshold=85,                                   # Number from 1-255, anything less than this number means there is tissue
    training_files=None                             # Training file for GNB-based detection
    )

# Create the extractor object
patch_extractor = PatchExtractor(tissue_detector,
    parameters,
    feature_map=fm,                       # See note below
    annotations=None                        # Object of Annotation Class (see other note below)
    )


patch_num = patch_extractor.extract(wsi_fn)
print("Done")

# if __name__ == "__main__":
#     # Run the extraction process
#     multiprocessing.set_start_method('spawn')
#     pool = multiprocessing.Pool(processes=num_processors)
#     pool.map(patch_extractor.extract, [wsi_fn])























