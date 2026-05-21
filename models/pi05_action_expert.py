from kernels.gated_matmul import matmul_small_gate
from kernels.attention import matmul_abT_scale, softmax_kernel_prefix_suffix
from ops.action_expert_ops import matmul_k8_n_256
from ops.action_expert_ops_pi05 import (
    matmul_k_32_1024_bias,
    matmul_k_1024_2560_qkv_rope,
    adarms_norm_style_proj,
    adarms_matmul_k_1024_32_bias_res,
    matmul_k_2048_1024_gate,
    matmul_k_4096_1024_gate,
)


def action_expert(weights, buffers, encoder_seq_len, num_steps=10):
    for step in range(num_steps):
        matmul_k_32_1024_bias(
            buffers['diffusion_noise'],
            weights['decoder_action_in_proj_w'],
            weights['decoder_action_in_proj_b'],
            buffers['decoder_x']
        )
        seq_len = buffers['decoder_x'].shape[0]
        for i in range(18):
            adarms_norm_style_proj(
                buffers['decoder_x'],
                buffers['decoder_time_emb'][step],
                weights['decoder_pre_attn_norm_mod_w'][i],
                weights['decoder_pre_attn_norm_mod_b'][i],
                buffers['x_normed_buf'],
                buffers['gate_buf'],
                buffers['decoder_style_attn'][step, i]
            )
            matmul_k_1024_2560_qkv_rope(
                buffers['x_normed_buf'],
                weights['decoder_attn_qkv_w'][i],
                buffers['decoder_rope_weights'],
                buffers['decoder_q_buf'],
                buffers['encoder_K'][i, encoder_seq_len:encoder_seq_len + seq_len],
                buffers['encoder_V'][i, encoder_seq_len:encoder_seq_len + seq_len],
            )
            total_queries = buffers['decoder_q_buf'].shape[0]
            prefix_keys = encoder_seq_len
            suffix_keys = seq_len
            total_keys = prefix_keys + suffix_keys

            matmul_abT_scale[(((total_queries + 31) // 32) * ((total_keys + 31) // 32),)](
                buffers['decoder_q_buf'],
                buffers['encoder_K'][i, :encoder_seq_len + seq_len],
                buffers['decoder_logits_buf'],
                total_queries,
                total_keys,
                256,
                256 ** -0.5,
                BLOCK_SIZE_M=32,
                BLOCK_SIZE_N=32,
                BLOCK_SIZE_K=64,
            )

            softmax_kernel_prefix_suffix[((total_queries + 3) // 4,)](
                buffers['decoder_logits_buf'],
                total_queries,
                prefix_keys,
                suffix_keys,
                buffers['valid_encoder_len'],
                buffers['decoder_attn_buf'],
                BLOCK_SIZE_M=4,
                BLOCK_SIZE=1024,
            )

            matmul_k8_n_256(
                buffers['decoder_attn_buf'],
                buffers['encoder_V'][i, :encoder_seq_len + seq_len],
                buffers['decoder_q_buf'],
            )
            matmul_k_2048_1024_gate(
                buffers['decoder_q_buf'].view(-1, 2048),
                weights['decoder_attn_o_w'][i],
                buffers['decoder_x'],
                buffers['gate_buf']
            )
            adarms_norm_style_proj(
                buffers['decoder_x'],
                buffers['decoder_time_emb'][step],
                weights['decoder_pre_ffn_norm_mod_w'][i],
                weights['decoder_pre_ffn_norm_mod_b'][i],
                buffers['x_normed_buf'],
                buffers['gate_buf'],
                buffers['decoder_style_ffn'][step, i]
            )
            seq_len = buffers['decoder_x'].shape[0]
            matmul_small_gate[( (seq_len + 127) // 128, (4096 + 63) // 64 )](
                buffers['x_normed_buf'],
                weights['decoder_ffn_gate_w'][i],
                weights['decoder_ffn_up_w'][i],
                buffers['decoder_hidden'],
                seq_len,
                1024,
                4096,
            )
            matmul_k_4096_1024_gate(
                buffers['decoder_hidden'],
                weights['decoder_ffn_down_w'][i],
                buffers['decoder_x'],
                buffers['gate_buf']
            )

        adarms_matmul_k_1024_32_bias_res(
            buffers['decoder_x'],
            buffers['decoder_time_emb'][step],
            weights['decoder_final_norm_mod_w'],
            weights['decoder_final_norm_mod_b'],
            buffers['x_normed_buf'],
            buffers['gate_buf'],
            buffers['decoder_style_final'][step],
            weights['decoder_action_out_proj_w'],
            weights['decoder_action_out_proj_b'],
            buffers['diffusion_noise'],
            buffers['diffusion_noise'],
        )
