
##############  different patch size ##################
# patch_size 256x256, 384x384, 512x512, 1024x1024, 2048x2048, 4096x4096
# Timing 40x, average file size
output_root=/lus/grand/projects/gpu_hack/mayopath/Jun/data
mkdir $output_root/512_64
python -m cProfile -o $output_root/512_64/512_64_s.stats patch_extraction_p_args_single.py -s 512 -o $output_root/512_64 -n 1 >> $output_root/log_512_64_s.txt





