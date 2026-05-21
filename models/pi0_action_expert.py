from ops.action_expert_ops import (
    matmul_1_32_1024_bias,
    matmul_k_32_1024_bias_silu,
    matmul_k_1024_1024_bias,
    rms_matmul_k_1024_2560_qkv_rope,
    matmul_k8_256_n_softmax_mask0,
    matmul_k8_n_256,
    matmul_k_2048_1024_res,
    rms_matmul_k_1024_4096_gate,
    matmul_k_4096_1024_res,
    rms_matmul_k_1024_32_bias_res,
)


def action_expert(weights, buffers, encoder_seq_len):
    matmul_1_32_1024_bias(
        buffers['observation_state_normalized'],
        weights['decoder_state_in_proj_w'],
        weights['decoder_state_in_proj_b'],
        buffers['decoder_state_buf']
    )
    for step in range(10):
        buffers['decoder_x'][:1].copy_(buffers['decoder_state_buf'])
        matmul_k_32_1024_bias_silu(
            buffers['diffusion_noise'],
            weights['decoder_action_fused_in_proj_w'],
            weights['decoder_action_fused_time_biases'][step%10],
            buffers['decoder_x_buf']
        )
        matmul_k_1024_1024_bias(
            buffers['decoder_x_buf'],
            weights['decoder_action_mlp_w'],
            weights['decoder_action_mlp_b'],
            buffers['decoder_x'][1:]
        )
        for i in range(18):
            rms_matmul_k_1024_2560_qkv_rope(
                buffers['decoder_x'], weights['decoder_attn_qkv_w'][i],
                buffers['decoder_rope_weights'],
                buffers['decoder_q_buf'],
                buffers['encoder_K'][i][encoder_seq_len:],
                buffers['encoder_V'][i][encoder_seq_len:],
                buffers['decoder_norm_factor_buf']
            )
            matmul_k8_256_n_softmax_mask0(
                buffers['decoder_q_buf'],
                buffers['encoder_K'][i],
                buffers['decoder_attn_buf'],
                encoder_seq_len
            )
            matmul_k8_n_256(
                buffers['decoder_attn_buf'],
                buffers['encoder_V'][i],
                buffers['decoder_q_buf']
            )

            matmul_k_2048_1024_res(
                buffers['decoder_q_buf'].view(-1, 2048),
                weights['decoder_attn_o_w'][i],
                buffers['decoder_x']
            )
            rms_matmul_k_1024_4096_gate(
                buffers['decoder_x'],
                weights['decoder_ffn_gate_w'][i],
                weights['decoder_ffn_up_w'][i],
                buffers['decoder_hidden'],
                buffers['decoder_norm_factor_buf']
            )
            matmul_k_4096_1024_res(
                buffers['decoder_hidden'],
                weights['decoder_ffn_down_w'][i],
                buffers['decoder_x']
            )
        rms_matmul_k_1024_32_bias_res(
            buffers['decoder_x'][1:],
            weights['decoder_action_fused_out_proj_w'],
            weights['decoder_action_fused_out_proj_b'],
            buffers['diffusion_noise'],
            buffers['decoder_norm_factor_buf'],
        )
