from kernels.matmul import (
    matmul_small,
    matmul_small_bias,
    matmul_small_bias_silu,
    matmul_small_res,
    matvec_bias_kernel,
    scaled_matmul_small_bias_res,
)
from kernels.gated_matmul import scaled_matmul_small_gate
from kernels.norm import rmsnorm_factor_kernel
from kernels.rope_qkv import scaled_matmul_rope_qkv
from kernels.attention import matmul_abT_scale, softmax_kernel_mask0


def matmul_1_32_1024_bias(x, weight, bias, out):
    matvec_bias_kernel[((1024 + 7)//8, )](
        x,
        weight,
        bias,
        out,
        features = 32,
        hidden = 1024
    )


def matmul_k_32_1024_bias_silu(x, weight, bias, out):
    seq_len = x.shape[0]
    matmul_small_bias_silu[((seq_len + 31) // 32) * (1024 // 32),] (
        x, weight, out, bias,
        seq_len = seq_len,
        features = 32,
        hidden = 1024,
        BLOCK_SIZE_N = 32,
        BLOCK_SIZE_M = 32,
        BLOCK_SIZE_K = 32
    )


def matmul_k_1024_1024_bias(x, weight, bias, out):
    seq_len = x.shape[0]
    matmul_small_bias[((seq_len + 31) // 32) * (1024 // 32),] (
        x, weight, out, bias,
        seq_len = seq_len,
        features = 1024,
        hidden = 1024,
        BLOCK_SIZE_N = 32,
        BLOCK_SIZE_M = 32,
        BLOCK_SIZE_K = 64
    )


def rms_matmul_k_1024_32_bias_res(x, weight, bias, out, x_norm_factor):
    seq_len = x.shape[0]
    rmsnorm_factor_kernel[(seq_len,)](x, x_norm_factor, seq_len, 1024, eps=1e-6, BLOCK_SIZE=1024)
    scaled_matmul_small_bias_res[((seq_len + 15) // 16) * (32 // 16),] (
        x, x_norm_factor, weight, out, bias, out,
        seq_len = seq_len,
        features = 1024,
        hidden = 32,
        BLOCK_SIZE_N = 16,
        BLOCK_SIZE_M = 16,
        BLOCK_SIZE_K = 256
    )


def rms_matmul_k_1024_2560_qkv_rope(x, weight_qkv, rope_weight, Q, K, V, x_norm_factor):
    seq_len = x.shape[0]
    rmsnorm_factor_kernel[(128,)](x, x_norm_factor, seq_len, 1024, eps=1e-6, BLOCK_SIZE=1024)
    scaled_matmul_rope_qkv[(128,)](
        x, x_norm_factor, seq_len, 1024, 256, 8,
        weight_qkv, rope_weight, Q, K, V,
    )


def matmul_k8_256_n_softmax_mask0(Q, K, out, encoder_seq_len):
    total_queries = Q.shape[0]
    total_keys = K.shape[0]
    head_dim = 256
    matmul_abT_scale[(((total_queries + 31) // 32) * ((total_keys + 31) // 32),)](Q, K, out,
        total_queries, total_keys, head_dim, head_dim ** -0.5,
        BLOCK_SIZE_M=32, BLOCK_SIZE_N=32, BLOCK_SIZE_K=64)
    softmax_kernel_mask0[((total_queries + 3) // 4,)](out,
        total_queries, total_keys, 8, encoder_seq_len, out,
        BLOCK_SIZE_M=4, BLOCK_SIZE=1024)


def matmul_k8_n_256(x, V, out):
    total_queries = x.shape[0]
    total_keys = V.shape[0]
    head_dim = 256
    matmul_small[((total_keys + 31) // 32) * (head_dim // 32),](
        x, V, out,
        seq_len = total_queries,
        features = total_keys,
        hidden = head_dim,
        BLOCK_SIZE_N = 32,
        BLOCK_SIZE_M = 32,
        BLOCK_SIZE_K = 64
    )


def matmul_k_2048_1024_res(x, weight, out):
    seq_len = x.shape[0]
    matmul_small_res[(128,)](
        x,
        weight,
        out,
        out,
        seq_len = seq_len,
        features = 2048,
        hidden = 1024,
        BLOCK_SIZE_N = 32,
        BLOCK_SIZE_M = 32,
        BLOCK_SIZE_K = 128
    )


def rms_matmul_k_1024_4096_gate(x, weight1, weight2, out, x_norm_factor):
    seq_len = x.shape[0]
    rmsnorm_factor_kernel[(128,)](x, x_norm_factor, seq_len, 1024, eps=1e-6, BLOCK_SIZE=1024)
    scaled_matmul_small_gate[(128,)] (
        x, x_norm_factor, weight1, weight2, out,
        seq_len = seq_len,
        features = 1024,
        hidden = 4096,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_K = 64
    )


def matmul_k_4096_1024_res(x, weight, out):
    seq_len = x.shape[0]
    matmul_small_res[(((seq_len + 15) // 16) * (1024 // 32),)](
        x,
        weight,
        out,
        out,
        seq_len = seq_len,
        features = 4096,
        hidden = 1024,
        BLOCK_SIZE_N = 16,
        BLOCK_SIZE_M = 32,
        BLOCK_SIZE_K = 256
    )
