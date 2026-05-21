from kernels.matmul import (
    matmul_small_bias,
    matmul_small_bias_res,
    matmul_small_bias_silu,
)
from kernels.gated_matmul import matmul_small_res_gate
from kernels.norm import adarms_norm_kernel
from kernels.rope_qkv import matmul_rope_qkv


def matmul_k_32_1024_bias(x, weight, bias, out):
    seq_len = x.shape[0]
    matmul_small_bias[((seq_len + 31) // 32) * (1024 // 32),] (
        x, weight, out, bias,
        seq_len = seq_len,
        features = 32,
        hidden = 1024,
        BLOCK_SIZE_N = 32,
        BLOCK_SIZE_M = 32,
        BLOCK_SIZE_K = 32
    )


def matmul_1_1024_1024_bias_silu(x, weight, bias, out):
    seq_len = x.shape[0]
    matmul_small_bias_silu[((seq_len + 31) // 32) * (1024 // 32),] (
        x, weight, out, bias,
        seq_len = seq_len,
        features = 1024,
        hidden = 1024,
        BLOCK_SIZE_N = 32,
        BLOCK_SIZE_M = 32,
        BLOCK_SIZE_K = 64
    )


def matmul_k_1024_2560_qkv_rope(x_normed, weight_qkv, rope_weight, Q, K, V):
    seq_len = x_normed.shape[0]
    matmul_rope_qkv[(128,)](
        x_normed, seq_len, 1024, 256, 8,
        weight_qkv, rope_weight, Q, K, V,
    )


def adarms_norm_style_proj(x, time_emb, mod_w, mod_b, x_normed, gate, style):
    seq_len = x.shape[0]
    adarms_norm_kernel[(seq_len,)](
        x,
        style,
        x_normed,
        gate,
        seq_len = seq_len,
        features = 1024,
        BLOCK_SIZE = 512
    )


def adarms_norm_style_proj_final(x, time_emb, mod_w, mod_b, x_normed, gate, style):
    seq_len = x.shape[0]

    adarms_norm_kernel[(seq_len,)](
        x,
        style,
        x_normed,
        gate,
        seq_len = seq_len,
        features = 1024,
        BLOCK_SIZE = 512
    )


def adarms_matmul_k_1024_32_bias_res(
    x,
    time_emb,
    mod_w,
    mod_b,
    x_normed,
    gate,
    style,
    weight,
    bias,
    out,
    res,
):
    adarms_norm_style_proj_final(x, time_emb, mod_w, mod_b, x_normed, gate, style)
    seq_len = x.shape[0]
    matmul_small_bias_res[((seq_len + 15) // 16) * (32 // 16),] (
        x_normed,
        weight,
        out,
        bias,
        res,
        seq_len = seq_len,
        features = 1024,
        hidden = 32,
        BLOCK_SIZE_N = 16,
        BLOCK_SIZE_M = 16,
        BLOCK_SIZE_K = 256
    )


def matmul_k_2048_1024_gate(x, weight, out, gate):
    seq_len = x.shape[0]
    matmul_small_res_gate[(128,)](
        x,
        weight,
        out,
        out,
        gate,
        seq_len = seq_len,
        features = 2048,
        hidden = 1024,
        BLOCK_SIZE_N = 32,
        BLOCK_SIZE_M = 32,
        BLOCK_SIZE_K = 128
    )


def matmul_k_4096_1024_gate(x, weight, out, gate):
    seq_len = x.shape[0]
    matmul_small_res_gate[(((seq_len + 15) // 16) * (1024 // 32),)](
        x,
        weight,
        out,
        out,
        gate,
        seq_len = seq_len,
        features = 4096,
        hidden = 1024,
        BLOCK_SIZE_N = 16,
        BLOCK_SIZE_M = 32,
        BLOCK_SIZE_K = 256
    )
