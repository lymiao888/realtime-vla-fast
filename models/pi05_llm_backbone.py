from ops.llm_ops import (
    layer_norm_matmul_n256_1152_2048_bias,
    rms_matmul_n_2048_2560_qkv_rope,
    matmul_n_2048_2048_res,
    rms_matmul_n_2048_16384_gate,
    matmul_n_16384_2048_res,
)
from ops.action_expert_ops import matmul_k8_n_256
from kernels.attention import matmul_abT_scale, softmax_kernel_masklen


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
            total_queries = buffers['encoder_Q'].shape[0]
            total_keys = encoder_seq_len
            matmul_abT_scale[(((total_queries + 31) // 32) * ((total_keys + 31) // 32),)](
                buffers['encoder_Q'],
                buffers['encoder_K'][i, :encoder_seq_len],
                buffers['encoder_logits_buf'],
                total_queries,
                total_keys,
                256,
                scale,
                BLOCK_SIZE_M=32,
                BLOCK_SIZE_N=32,
                BLOCK_SIZE_K=64,
            )
            softmax_kernel_masklen[((total_queries + 3) // 4,)](
                buffers['encoder_logits_buf'],
                total_queries,
                total_keys,
                buffers['valid_encoder_len'],
                buffers['encoder_attn_buf'],
                BLOCK_SIZE_M=4,
                BLOCK_SIZE=1024,
            )
            matmul_k8_n_256(
                buffers['encoder_attn_buf'],
                buffers['encoder_V'][i, :encoder_seq_len],
                buffers['encoder_ctx_buf'],
            )

            matmul_n_2048_2048_res(
                buffers['encoder_ctx_buf'].view(-1, 2048),
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
