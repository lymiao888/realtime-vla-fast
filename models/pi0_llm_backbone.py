from ops.llm_ops import (
    layer_norm_matmul_n256_1152_2048_bias,
    rms_matmul_n_2048_2560_qkv_rope,
    AttnSingleKey,
    matmul_n_2048_2048_res,
    rms_matmul_n_2048_16384_gate,
    matmul_n_16384_2048_res,
)


def llm_backbone(weights, buffers, encoder_seq_len):
    layer_norm_matmul_n256_1152_2048_bias(
        buffers['vision_x'],
        weights['vision_final_norm_w'],
        weights['vision_final_norm_b'],
        weights['encoder_multi_modal_projector_w'],
        weights['encoder_multi_modal_projector_b'],
        buffers['encoder_x'],
        buffers['vision_x_norm']
    )
    for i in range(18):
        rms_matmul_n_2048_2560_qkv_rope(
            buffers['encoder_x'],
            weights['encoder_attn_qkv_w'][i],
            buffers['encoder_rope_weights'],
            buffers['encoder_Q'],
            buffers['encoder_K'][i, :encoder_seq_len],
            buffers['encoder_V'][i, :encoder_seq_len],
            buffers['encoder_x_norm']
        )

        if i != 17:
            scale = 1.0 / (256 ** 0.5)
            attn = AttnSingleKey(buffers['encoder_Q'], buffers['encoder_K'][i, :encoder_seq_len], buffers['encoder_V'][i, :encoder_seq_len], scale)

            matmul_n_2048_2048_res(
                attn,
                weights['encoder_attn_o_w'][i],
                buffers['encoder_x']
            )

            rms_matmul_n_2048_16384_gate(
                buffers['encoder_x'],
                weights['encoder_ffn_gate_w'][i],
                weights['encoder_ffn_up_w'][i],
                buffers['encoder_hidden'],
                buffers['encoder_x_norm']
            )

            matmul_n_16384_2048_res(
                buffers['encoder_hidden'],
                weights['encoder_ffn_down_w'][i],
                buffers['encoder_x']
            )
