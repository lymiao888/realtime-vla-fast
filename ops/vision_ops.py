import torch

from kernels.matmul import (
    matmul_small_bias,
    matmul_small_bias_res,
    matmul_small_bias_res_mod,
    matmul_small_bias_gelu,
    matmul_split_k,
    merge_split_k_bias_res,
    matmul_512x1152x1152_twopart_bias_res,
    combine_1536_1152_twopart,
)
from kernels.norm import layer_norm_small_kernel


def conv2d_embed_n256_1152_res(images, patch_w, patch_b, pos_emb, out):
    nviews = images.shape[0]
    img_input = images.view(nviews, 16, 14, 16, 14, 3).permute(0, 1, 3, 2, 4, 5).contiguous()
    matmul_small_bias_res_mod[(256 * nviews // 64) * (1152 // 64),](
        img_input,
        patch_w,
        out,
        patch_b,
        pos_emb,
        seq_len = 256 * nviews,
        features = 3 * 14 * 14,
        hidden = 1152,
        i_mod = 256,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_K = 32
    )


def layer_norm_n256_1152(x, norm_w, norm_b, out):
    num_views = x.shape[0]
    seq_len = 256 * num_views
    layer_norm_small_kernel[seq_len,](
        x,
        out,
        norm_w,
        norm_b,
        seq_len = seq_len,
        features = 1152
    )


def layer_norm_QKV_matmul_n256_1152_3456_bias(x, norm_w, norm_b, qkv_w, qkv_b, out, x_norm):
    num_views = x.shape[0]
    seq_len = 256 * num_views
    layer_norm_small_kernel[seq_len,](
        x,
        x_norm,
        norm_w,
        norm_b,
        seq_len = seq_len,
        features = 1152
    )

    matmul_small_bias[((seq_len + 63) // 64) * (3456 // 64),](
        x_norm,
        qkv_w,
        out,
        qkv_b,
        seq_len = seq_len,
        features = 1152,
        hidden = 3456,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_K = 32
    )


def matmul_n256_1152_1152_bias_res(x, weight, bias, res, out, buf):
    num_views = x.shape[0]
    if num_views == 2:
        matmul_512x1152x1152_twopart_bias_res[(256,)](
            old_ptr = res,
            inp_ptr = x,
            weight_ptr = weight,
            bias_ptr = bias,
            out_ptr = out,
            out2_ptr = buf,
            seq_len = 512,
            features = 1152,
            hidden = 1152,
        )
        combine_1536_1152_twopart[(256,)](
            inp_ptr = buf,
            out_ptr = out,
            seq_len = 512,
            hidden = 1152,
        )
        return
    seq_len = 256 * num_views
    matmul_small_bias_res[((seq_len + 31) // 32) * (1152 // 64),](
        x,
        weight,
        out,
        bias,
        res,
        seq_len = seq_len,
        features = 1152,
        hidden = 1152,
        BLOCK_SIZE_N = 32,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_K = 32
    )


def layer_norm_matmul_n256_1152_4304_bias_gelu(x, norm_w, norm_b, weight, bias, out, x_norm):
    num_views = x.shape[0]
    seq_len = 256 * num_views

    layer_norm_small_kernel[seq_len,](
        x,
        x_norm,
        norm_w,
        norm_b,
        seq_len = seq_len,
        features = 1152
    )

    BLOCK_SIZE_N = 64
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_K = 64
    matmul_small_bias_gelu[((seq_len + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N) * ((4304 + (BLOCK_SIZE_M - 1)) // BLOCK_SIZE_M),](
        x_norm,
        weight,
        out,
        bias,
        seq_len = seq_len,
        features = 1152,
        hidden = 4304,
        BLOCK_SIZE_N = BLOCK_SIZE_N,
        BLOCK_SIZE_M = BLOCK_SIZE_M,
        BLOCK_SIZE_K = BLOCK_SIZE_K
    )


def matmul_n256_4304_1152_bias_res(x, weight, bias, res, out, buf):
    num_views = x.shape[0]
    seq_len = 256 * num_views
    matmul_split_k[((seq_len + 64) // 64) * (1152 // 64) * 4,](
        x,
        weight,
        buf,
        seq_len = seq_len,
        features = 4304,
        hidden = 1152,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_K = 64,
        SPLIT_K = 4
    )
    merge_split_k_bias_res[(seq_len * 1152 + 1023) // 1024,](
        buf,
        bias,
        res,
        out,
        seq_len = seq_len,
        hidden = 1152,
        SPLIT_K = 4
    )


@torch.compile
def AttnMultiKey(QKV):
    QKV = QKV.view(-1, 256, 3, 16, 72).permute(0, 2, 3, 1, 4)
    Q = QKV[:, 0]
    K = QKV[:, 1]
    V = QKV[:, 2]
    attn = torch.nn.functional.scaled_dot_product_attention(Q, K, V)
    attn = attn.transpose(1, 2).reshape(Q.shape[0], 256, 1152)
    return attn
