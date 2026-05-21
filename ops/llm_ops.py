import torch

from kernels.matmul import matmul_small_bias, matmul_small_res
from kernels.gated_matmul import matmul_small_gate
from kernels.norm import layer_norm_small_kernel, rms_norm_kernel
from kernels.rope_qkv import matmul_n_2048_2560_qkv_rope


def rms_matmul_n_2048_16384_gate(x, weight1, weight2, out, x_norm):
    seq_len = x.shape[0]
    rms_norm_kernel[(seq_len,)](x, x_norm, seq_len, 2048)
    matmul_small_gate[( (seq_len + 127)//128, (16384 + 63)//64 )](
        x_norm, weight1, weight2, out, seq_len,
        2048, 16384
    )


def matmul_n_16384_2048_res(x, weight, out):
    seq_len = x.shape[0]
    BLOCK_SIZE_N = 128
    if seq_len < 512:
        BLOCK_SIZE_N = 64
    matmul_small_res[((seq_len + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N) * (2048 // 64),](
        x,
        weight,
        out,
        out,
        seq_len = seq_len,
        features = 16384,
        hidden = 2048,
        BLOCK_SIZE_N = BLOCK_SIZE_N,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_K = 64
    )


def layer_norm_matmul_n256_1152_2048_bias(x, norm_w, norm_b, proj_w, proj_b, out, x_norm):
    seq_len = x.shape[0] * 256
    layer_norm_small_kernel[seq_len,](
        x,
        x_norm,
        norm_w,
        norm_b,
        seq_len = seq_len,
        features = 1152,
        eps = 1e-5
    )
    matmul_small_bias[((seq_len + 63) // 64) * (2048 // 64),](
        x_norm,
        proj_w,
        out,
        proj_b,
        seq_len = seq_len,
        features = 1152,
        hidden = 2048,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_K = 64
    )


def rms_matmul_n_2048_2560_qkv_rope(x, weight_qkv, rope_weight, Q, K, V, x_norm):
    seq_len = x.shape[0]
    rms_norm_kernel[(seq_len,)](x, x_norm, seq_len, 2048)
    matmul_n_2048_2560_qkv_rope[((seq_len + 63) // 64, 2560 // 64)](
        x_norm, weight_qkv, rope_weight, Q, K, V, seq_len, 2048, 256, 8
    )


@torch.compile
def AttnSingleKey(Q, K, V, scale):
    logits = torch.matmul(Q, K.T) * scale
    logits = torch.nn.functional.softmax(logits, dim=-1)
    attn = torch.matmul(logits, V).view(-1, 2048)
    return attn


def matmul_n_2048_2048_res(x, weight, out):
    seq_len = x.shape[0]
    matmul_small_res[((seq_len + 127) // 128) * (2048 // 64),](
        x,
        weight,
        out,
        out,
        seq_len = seq_len,
        features = 2048,
        hidden = 2048,
        BLOCK_SIZE_N = 128,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_K = 64
    )
