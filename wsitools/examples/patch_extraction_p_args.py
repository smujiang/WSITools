from wsitools.tissue_detection.tissue_detector import TissueDetector
from wsitools.patch_extraction.patch_extractor import ExtractorParameters, PatchExtractor
import multiprocessing
import os
import argparse
import csv
import random

parser = argparse.ArgumentParser()

parser.add_argument("-o", "--out-dir",
                    default=os.getcwd(),
                    dest='out_dir',
                    help="Where patches should be saved")

parser.add_argument("-s", "--patch-size",
                    default=256,
                    dest='patch_size',
                    type=int,
                    help="H & W of patches")

parser.add_argument("-n", "--number-processors",
                    default=8,
                    dest='num_processors',
                    type=int,
                    help="Number of processors to use during patch extraction")

parser.add_argument("-R", "--rescale-rate",
                    default=128,
                    dest='rescale_rate',
                    type=int,
                    help="Fold size to scale the thumbnail to (for faster processing)")

parser.add_argument("-f", "--patch-format",
                    dest='save_format',
                    choices=['.png', '.jpg', '.h5', '.tfrecord'],
                    default=".h5",
                    help="Output format for patches")

parser.add_argument("-l", "--openslide-level",
                    dest='openslide_level',
                    default=0,
                    help="Level used to extract patches")

args = parser.parse_args()

wsi_fn_list_csv = "./wsi_list_40x.csv"
output_dir = args.out_dir
log_dir = os.path.join(args.out_dir, "logs")
patch_size = args.patch_size
num_processors = args.num_processors

save_format = args.save_format
openslide_level = args.openslide_level
rescale_rate = args.rescale_rate


fp = open(wsi_fn_list_csv, 'r')
wsi_fn_list = [i.strip() for i in fp.readlines()]
wsi_fn_list = random.choices(wsi_fn_list, k=10)

# wsi_fn_list = [os.path.join("/infodev1/non-phi-data/junjiang/OvaryCancer/WSIs", "OCMC-{:03d}.svs".format(i)) for i in range(1, 31)]
#
# output_dir = "/infodev1/non-phi-data/junjiang/OvaryCancer/Patches/h5_files"
#
# log_dir = "/infodev1/non-phi-data/junjiang/OvaryCancer/Patches/logs"
#


tissue_detector = TissueDetector("LAB_Threshold", threshold=85)  #

parameters = ExtractorParameters(output_dir, log_dir=log_dir, patch_size=patch_size, stride=patch_size, extract_layer=0, save_format='.h5', sample_cnt=-1)

patch_extractor = PatchExtractor(tissue_detector, parameters=parameters)


pool = multiprocessing.Pool(processes=num_processors)
pool.map(patch_extractor.extract, wsi_fn_list)




