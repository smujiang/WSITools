
##############  different patch size ##################
# patch_size 256x256, 384x384, 512x512, 1024x1024, 2048x2048, 4096x4096
# Timing 40x, average file size
output_root=/lus/grand/projects/gpu_hack/mayopath/Jun/data
mkdir $output_root/256_64
python -m cProfile -o $output_root/256_64/256_64.stats patch_extraction_p_args.py -s 256 -o $output_root/256_64 -n 64 >> $output_root/log_256_64.txt

mkdir $output_root/384_64
python -m cProfile -o $output_root/384_64/384_64.stats patch_extraction_p_args.py -s 384 -o $output_root/384_64 -n 64 >> $output_root/log_384_64.txt

mkdir $output_root/512_64
python -m cProfile -o $output_root/512_64/512_64.stats patch_extraction_p_args.py -s 512 -o $output_root/512_64 -n 64 >> $output_root/log_512_64.txt

mkdir $output_root/1024_64
python -m cProfile -o $output_root/1024_64/1024_64.stats patch_extraction_p_args.py -s 1024 -o $output_root/1024_64 -n 64 >> $output_root/log_1024_64.txt

mkdir $output_root/2048_64
python -m cProfile -o $output_root/2048_64/2048_64.stats patch_extraction_p_args.py -s 2048 -o $output_root/2048_64 -n 64 >> $output_root/log_2048_64.txt

############## different number of process ##################
mkdir $output_root/512_8
python -m cProfile -o $output_root/512_8/512_8.stats patch_extraction_p_args.py -s 512 -o $output_root/512_8 -n 8 >> $output_root/log_512_8.txt

mkdir $output_root/512_16
python -m cProfile -o $output_root/512_16/512_16.stats patch_extraction_p_args.py -s 512 -o $output_root/512_16 -n 16 >> $output_root/log_512_16.txt

mkdir $output_root/512_32
python -m cProfile -o $output_root/512_32/512_32.stats patch_extraction_p_args.py -s 512 -o $output_root/512_32 -n 32 >> $output_root/log_512_32.txt

mkdir $output_root/512_128
python -m cProfile -o $output_root/512_128/512_128.stats patch_extraction_p_args.py -s 512 -o $output_root/512_128 -n 128 >> $output_root/log_512_128.txt



################################
# chunking, before and after, average file size
#



