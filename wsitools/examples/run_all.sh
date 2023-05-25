
##############  different patch size ##################
# patch_size 256x256, 384x384, 512x512, 1024x1024, 2048x2048, 4096x4096
# Timing 40x, average file size
output_root=/lus/grand/projects/gpu_hack/mayopath/Jun/data

for size in 256 384 512 1024 2048
do
  for n_p in 8 16 32 64 128
  do
    mkdir $output_root/"$size"_"$n_p"
    python -m cProfile -o $output_root/"$size"_"$n_p"/"$size"_"$n_p".stats patch_extraction_p_args.py -s $size -o $output_root/"$size"_"$n_p" -n $n_p >> $output_root/log_"$size"_"$n_p".txt
  done
done



################################
# chunking, before and after, average file size
#



