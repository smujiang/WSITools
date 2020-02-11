from wsitools.tissue_detection.tissue_detector import TissueDetector
from wsitools.patch_extraction.patch_extractor import ExtractorParameters, PatchExtractor
import multiprocessing

#Define some run parameters
num_processors = 20                     # Number of processes that can be running at once
wsi_fn = ["/path/2/file.tff", ""]             # Define a sample image that can be read by OpenSlide
output_dir = "/data/WSIs_extraction"    # Define an output directory

# Define the parameters for Patch Extraction, including generating an thumbnail from which to traverse over to find
# tissue.
parameters = ExtractorParameters(output_dir, # Where the patches should be extracted to
    save_format='.png',                      # Can be '.jpg', '.png', or '.tfrecord'
    sample_cnt=-1,                           # Limit the number of patches to extract (-1 == all patches)
    patch_size=128,                          # Size of patches to extract (Height & Width)
    rescale_rate=128,                        # Fold size to scale the thumbnail to (for faster processing)
    patch_filter_by_area=0.5,                # Amount of tissue that should be present in a patch
    with_anno=True,                          # If true, you need to supply an additional XML file
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
    feature_map=None,                       # See note below
    annotations=None                        # Object of Annotation Class (see other note below)
    )

# Run the extraction process
multiprocessing.set_start_method('spawn')
pool = multiprocessing.Pool(processes=num_processors)
pool.map(patch_extractor.extract, [wsi_fn])







