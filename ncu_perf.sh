ncu --target-processes application-only \
    --nvtx \
    --nvtx-include "matmul_small_16384_2048_res]" \
    --set full \
    --kernel-name-base demangled \
    --launch-count 20 \
    -o ncu_pi05_matmul_small_16384_2048_res_opt2 \
    python benchmark.py --model_version pi05 --num_views 3 --chunk_size 50 --prompt_len 10