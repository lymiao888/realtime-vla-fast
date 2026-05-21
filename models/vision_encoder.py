from ops.vision_ops import (
    conv2d_embed_n256_1152_res,
    layer_norm_QKV_matmul_n256_1152_3456_bias,
    matmul_n256_1152_1152_bias_res,
    layer_norm_matmul_n256_1152_4304_bias_gelu,
    matmul_n256_4304_1152_bias_res,
    AttnMultiKey,
)


def vision_encoder(weights, buffers, num_views):
    conv2d_embed_n256_1152_res(
        buffers['observation_images_normalized'],
        weights['vision_patch_embedding_w'],
        weights['vision_patch_embedding_b'],
        weights['vision_position_embedding'],
        buffers['vision_x']
    )

    for i in range(27):
        layer_norm_QKV_matmul_n256_1152_3456_bias(
            buffers['vision_x'],
            weights['vision_pre_attn_norm_w'][i],
            weights['vision_pre_attn_norm_b'][i],
            weights['vision_attn_qkv_w'][i],
            weights['vision_attn_qkv_b'][i],
            buffers['vision_QKV'],
            buffers['vision_x_norm']
        )

        attn = AttnMultiKey(buffers['vision_QKV'])

        matmul_n256_1152_1152_bias_res(
            attn,
            weights['vision_attn_o_w'][i],
            weights['vision_attn_o_b'][i],
            buffers['vision_x'],
            buffers['vision_x'],
            buffers['vision_x_split_k_buf']
        )

        layer_norm_matmul_n256_1152_4304_bias_gelu(
            buffers['vision_x'],
            weights['vision_pre_ffn_norm_w'][i],
            weights['vision_pre_ffn_norm_b'][i],
            weights['vision_ffn_up_w'][i],
            weights['vision_ffn_up_b'][i],
            buffers['vision_hidden'],
            buffers['vision_x_norm']
        )

        matmul_n256_4304_1152_bias_res(
            buffers['vision_hidden'],
            weights['vision_ffn_down_w'][i],
            weights['vision_ffn_down_b'][i],
            buffers['vision_x'],
            buffers['vision_x'],
            buffers['vision_x_split_k_buf']
        )
