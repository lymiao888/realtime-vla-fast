import torch

from models.vision_encoder import vision_encoder
from models.pi0_llm_backbone import llm_backbone
from models.pi0_action_expert import action_expert


def pi0_model(weights, buffers, num_views):
    encoder_seq_len = buffers['encoder_x'].shape[0]
    vision_encoder(weights, buffers, num_views)
    llm_backbone(weights, buffers, encoder_seq_len)
    action_expert(weights, buffers, encoder_seq_len)


class Pi0Inference:
    def __init__(self, checkpoint, num_views, chunk_size):
        self.num_views = num_views
        self.chunk_size = chunk_size
        encoded_prompt = checkpoint['language_embeds']
        self.prompt_len = len(encoded_prompt)

        self.weights = {
            "vision_patch_embedding_w":           torch.empty(14, 14, 3, 1152,        dtype = torch.bfloat16, device = "cuda"),
            "vision_patch_embedding_b":           torch.empty(1152,                   dtype = torch.bfloat16, device = "cuda"),
            "vision_position_embedding":          torch.empty(256, 1152,              dtype = torch.bfloat16, device = "cuda"),
            "vision_attn_qkv_w":                  torch.empty(27, 1152, 3 * 1152,     dtype = torch.bfloat16, device = "cuda"),
            "vision_attn_qkv_b":                  torch.empty(27, 3 * 1152,           dtype = torch.bfloat16, device = "cuda"),
            "vision_attn_o_w":                    torch.empty(27, 1152, 1152,         dtype = torch.bfloat16, device = "cuda"),
            "vision_attn_o_b":                    torch.empty(27, 1152,               dtype = torch.bfloat16, device = "cuda"),
            "vision_ffn_up_w":                    torch.empty(27, 1152, 4304,         dtype = torch.bfloat16, device = "cuda"),
            "vision_ffn_up_b":                    torch.empty(27, 4304,               dtype = torch.bfloat16, device = "cuda"),
            "vision_ffn_down_w":                  torch.empty(27, 4304, 1152,         dtype = torch.bfloat16, device = "cuda"),
            "vision_ffn_down_b":                  torch.empty(27, 1152,               dtype = torch.bfloat16, device = "cuda"),
            "vision_pre_attn_norm_w":             torch.empty(27, 1152,               dtype = torch.bfloat16, device = "cuda"),
            "vision_pre_attn_norm_b":             torch.empty(27, 1152,               dtype = torch.bfloat16, device = "cuda"),
            "vision_pre_ffn_norm_w":              torch.empty(27, 1152,               dtype = torch.bfloat16, device = "cuda"),
            "vision_pre_ffn_norm_b":              torch.empty(27, 1152,               dtype = torch.bfloat16, device = "cuda"),
            "vision_final_norm_w":                torch.empty(1152,                   dtype = torch.bfloat16, device = "cuda"),
            "vision_final_norm_b":                torch.empty(1152,                   dtype = torch.bfloat16, device = "cuda"),

            "encoder_multi_modal_projector_w":    torch.empty(1152, 2048,             dtype = torch.bfloat16, device = "cuda"),
            "encoder_multi_modal_projector_b":    torch.empty(2048,                   dtype = torch.bfloat16, device = "cuda"),
            "encoder_attn_qkv_w":                 torch.empty(18, 2048, 2560,         dtype = torch.bfloat16, device = "cuda"),
            "encoder_attn_o_w":                   torch.empty(18, 2048, 2048,         dtype = torch.bfloat16, device = "cuda"),
            "encoder_ffn_gate_w":                 torch.empty(18, 2048, 16384,        dtype = torch.bfloat16, device = "cuda"),
            "encoder_ffn_up_w":                   torch.empty(18, 2048, 16384,        dtype = torch.bfloat16, device = "cuda"),
            "encoder_ffn_down_w":                 torch.empty(18, 16384, 2048,        dtype = torch.bfloat16, device = "cuda"),

            "decoder_state_in_proj_w":            torch.empty(32, 1024,               dtype = torch.bfloat16, device = "cuda"),
            "decoder_state_in_proj_b":            torch.empty(1024,                   dtype = torch.bfloat16, device = "cuda"),
            "decoder_action_fused_in_proj_w":     torch.empty(32, 1024,               dtype = torch.bfloat16, device = "cuda"),
            "decoder_action_fused_time_biases":   torch.empty(10, 1024,               dtype = torch.bfloat16, device = "cuda"),
            "decoder_action_mlp_w":               torch.empty(1024, 1024,             dtype = torch.bfloat16, device = "cuda"),
            "decoder_action_mlp_b":               torch.empty(1024,                   dtype = torch.bfloat16, device = "cuda"),
            "decoder_attn_qkv_w":                 torch.empty(18, 1024, 2560,         dtype = torch.bfloat16, device = "cuda"),
            "decoder_attn_o_w":                   torch.empty(18, 2048, 1024,         dtype = torch.bfloat16, device = "cuda"),
            "decoder_ffn_gate_w":                 torch.empty(18, 1024, 4096,         dtype = torch.bfloat16, device = "cuda"),
            "decoder_ffn_up_w":                   torch.empty(18, 1024, 4096,         dtype = torch.bfloat16, device = "cuda"),
            "decoder_ffn_down_w":                 torch.empty(18, 4096, 1024,         dtype = torch.bfloat16, device = "cuda"),
            "decoder_action_fused_out_proj_w":    torch.empty(1024, 32,               dtype = torch.bfloat16, device = "cuda"),
            "decoder_action_fused_out_proj_b":    torch.empty(32,                     dtype = torch.bfloat16, device = "cuda"),

            "language_embeds": torch.empty(self.prompt_len, 2048,  dtype = torch.bfloat16, device = "cuda"),
        }

        encoder_seq_len = num_views * 256 + self.prompt_len
        decoder_seq_len = chunk_size + 1

        self.buffers = {
            'observation_images_normalized':      torch.empty(num_views, 224, 224,3,           dtype=torch.bfloat16,   device = "cuda"),
            'observation_state_normalized':       torch.empty(32,                              dtype = torch.bfloat16, device = "cuda"),
            'diffusion_noise':                    torch.empty(chunk_size, 32,                  dtype = torch.bfloat16, device = "cuda"),
            'vision_x':                           torch.empty(num_views, 256, 1152,            dtype = torch.bfloat16, device = "cuda"),
            'vision_x_norm':                      torch.empty(num_views, 256, 1152,            dtype = torch.bfloat16, device = "cuda"),
            'vision_QKV':                         torch.empty(num_views, 256, 3 * 1152,        dtype = torch.bfloat16, device = "cuda"),
            'vision_hidden':                      torch.empty(num_views, 256, 4304,            dtype = torch.bfloat16, device = "cuda"),
            'vision_x_split_k_buf':               torch.empty((num_views * 256 * 1152 * 4,),   dtype = torch.float32, device = "cuda"),
            'encoder_rope_weights':               torch.empty(encoder_seq_len, 256,            dtype = torch.bfloat16, device = "cuda"),
            'encoder_x':                          torch.empty(encoder_seq_len, 2048,           dtype = torch.bfloat16, device = "cuda"),
            'encoder_x_norm':                     torch.empty(encoder_seq_len, 2048,           dtype = torch.bfloat16, device = "cuda"),
            'encoder_K':                          torch.empty(18, encoder_seq_len + decoder_seq_len, 256,   dtype = torch.bfloat16, device = "cuda"),
            'encoder_V':                          torch.empty(18, encoder_seq_len + decoder_seq_len, 256,   dtype = torch.bfloat16, device = "cuda"),
            'encoder_Q':                          torch.empty(encoder_seq_len * 8, 256,        dtype = torch.bfloat16, device = "cuda"),
            'encoder_hidden':                     torch.empty(encoder_seq_len, 16384,          dtype = torch.bfloat16, device = "cuda"),
            'decoder_rope_weights':               torch.empty(decoder_seq_len, 256,            dtype = torch.bfloat16, device = "cuda"),
            'decoder_x':                          torch.empty((decoder_seq_len, 1024),         dtype = torch.bfloat16, device = "cuda"),
            'decoder_x_buf':                      torch.empty((chunk_size, 1024),              dtype = torch.bfloat16, device = "cuda"),
            'decoder_state_buf':                  torch.empty((1, 1024),                       dtype = torch.bfloat16, device = "cuda"),
            'decoder_norm_factor_buf':            torch.empty((decoder_seq_len,),              dtype = torch.bfloat16, device = "cuda"),
            'decoder_q_buf':                      torch.empty((decoder_seq_len * 8, 256),      dtype = torch.bfloat16, device = "cuda"),
            'decoder_attn_buf':                   torch.empty((decoder_seq_len * 8, encoder_seq_len + decoder_seq_len),  dtype = torch.bfloat16, device = "cuda"),
            'decoder_hidden':                     torch.empty((decoder_seq_len, 4096),         dtype = torch.bfloat16, device = "cuda"),
            'decode_split_k_buf':                 torch.empty((2, decoder_seq_len, 1024),      dtype = torch.float32, device = "cuda"),
        }

        position_ids = torch.arange(encoder_seq_len, device="cuda")
        inv_freq = 1.0 / (10000 ** (torch.arange(0, 256, 2, dtype=torch.float32, device="cuda") / 256))
        k_phase = inv_freq[None, :] * position_ids[:, None]
        k_cos = torch.cos(k_phase).to(torch.bfloat16)
        k_sin = torch.sin(k_phase).to(torch.bfloat16)
        self.buffers['encoder_rope_weights'].copy_(
            torch.cat([k_cos[:, :, None], k_sin[:, :, None]], 2).view(-1, 256)
        )

        position_ids = torch.arange(decoder_seq_len, device="cuda") + encoder_seq_len
        inv_freq = 1.0 / (10000 ** (torch.arange(0, 256, 2, dtype=torch.float32, device="cuda") / 256))
        k_phase = inv_freq[None, :] * position_ids[:, None]
        k_cos = torch.cos(k_phase).to(torch.bfloat16)
        k_sin = torch.sin(k_phase).to(torch.bfloat16)
        self.buffers['decoder_rope_weights'].copy_(
            torch.cat([k_cos[:, :, None], k_sin[:, :, None]], 2).view(-1, 256)
        )

        for k, v in checkpoint.items():
            self.weights[k].copy_(v)

        self.infer_graph = torch.cuda.CUDAGraph()
        self.record_infer_graph()

    def record_run(self):
        self.buffers['encoder_x'][self.num_views * 256:].copy_(self.weights['language_embeds'])
        pi0_model(self.weights, self.buffers, self.num_views)

    def record_infer_graph(self):
        for i in range(3):
            self.record_run()
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            self.infer_graph.capture_begin()
            self.record_run()
            self.infer_graph.capture_end()

    def forward(self, observation_images_normalized, observation_state_normalized, diffusion_noise):
        self.buffers['observation_images_normalized'].copy_(observation_images_normalized)
        self.buffers['observation_state_normalized'].copy_(observation_state_normalized)
        self.buffers['diffusion_noise'].copy_(diffusion_noise)
        self.infer_graph.replay()
        return self.buffers['diffusion_noise']
