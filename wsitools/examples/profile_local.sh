# Configuration ------------------------

PROFILE_DIR="$MayoPath/Jun/data/test"
mkdir -p "$PROFILE_DIR"
PROFILE_OUT="$PROFILE_DIR/patch_extract_local_gpu.prf" # Change to set profiling differently

export PMI_LOCAL_RANK=1

# Set up environment
source "$MayoPath/init_mmae.sh"

(

# Nsys profile from https://dev-discuss.pytorch.org/t/using-nsight-systems-to-profile-gpu-workload/59
nsys profile -w true \
	-t cuda,nvtx,osrt,cudnn,cublas \
	-s cpu \
	--capture-range=cudaProfilerApi \
	--capture-range-end=stop \
	--cudabacktrace=true \
	-x true \
	-o $PROFILE_OUT \
python -m patch_extraction_p_args
	)