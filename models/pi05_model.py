import torch
import numpy as np
import torch.nn as nn
from transformers import AutoTokenizer

from kernels.matmul import matmul_small_bias

from models.vision_encoder import vision_encoder
from models.pi05_llm_backbone import llm_backbone
from models.pi05_action_expert import action_expert
from ops.action_expert_ops_pi05 import matmul_1_1024_1024_bias_silu


def pi05_model(weights, buffers, num_views, encoder_seq_len, num_steps=10):
    vision_encoder(weights, buffers, num_views)
    llm_backbone(weights, buffers, encoder_seq_len)
    action_expert(weights, buffers, encoder_seq_len, num_steps)


class Pi05Inference:
    def __init__(
        self,
        checkpoint,
        num_views,
        chunk_size,
        tokenizer_path: str | None = None,
        max_tokenize_len: int = 200,
        discrete_state_input: bool = True,
        max_prompt_text: str | None = None,
        state_dim_for_max_prompt: int | None = None,
    ):
        self.discrete_state_input = discrete_state_input
        self.tokenizer_path = tokenizer_path
        self.checkpoint = checkpoint
        self.num_views = num_views
        self.chunk_size = chunk_size
        self.max_tokenize_len = int(max_tokenize_len)
        if discrete_state_input:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
            if max_prompt_text is not None and state_dim_for_max_prompt is not None:
                self.max_prompt_len = self.estimate_max_prompt_len(
                    tokenizer=self.tokenizer,
                    task_prompt=max_prompt_text,
                    state_dim=int(state_dim_for_max_prompt),
                    max_tokenize_len=self.max_tokenize_len,
                    state_token_value=255,
                )
            else:
                self.max_prompt_len = self.max_tokenize_len
        else:
            self.max_prompt_len = len(checkpoint['language_embeds'])
        print(f"max_prompt_len: {self.max_prompt_len}, max_tokenize_len: {self.max_tokenize_len}")
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

            "decoder_time_embeds":                torch.zeros(10, 1024,                  dtype=torch.bfloat16, device="cuda"),
            "decoder_time_mlp_in_w":              torch.empty(1024, 1024,             dtype = torch.bfloat16, device = "cuda"),
            "decoder_time_mlp_in_b":              torch.empty(1024,                   dtype = torch.bfloat16, device = "cuda"),
            "decoder_time_mlp_out_w":             torch.empty(1024, 1024,             dtype = torch.bfloat16, device = "cuda"),
            "decoder_time_mlp_out_b":             torch.empty(1024,                   dtype = torch.bfloat16, device = "cuda"),
            "decoder_action_in_proj_w":           torch.empty(32, 1024,                      dtype = torch.bfloat16, device = "cuda"),
            "decoder_action_in_proj_b":           torch.empty(1024,                          dtype = torch.bfloat16, device = "cuda"),
            "decoder_pre_attn_norm_mod_w":        torch.empty(18, 1024, 3 * 1024,     dtype = torch.bfloat16, device = "cuda"),
            "decoder_pre_attn_norm_mod_b":        torch.empty(18, 3 * 1024,           dtype = torch.bfloat16, device = "cuda"),
            "decoder_pre_ffn_norm_mod_w":         torch.empty(18, 1024, 3 * 1024,     dtype = torch.bfloat16, device = "cuda"),
            "decoder_pre_ffn_norm_mod_b":         torch.empty(18, 3 * 1024,           dtype = torch.bfloat16, device = "cuda"),
            "decoder_attn_qkv_w":                 torch.empty(18, 1024, 2560,         dtype = torch.bfloat16, device = "cuda"),
            "decoder_attn_o_w":                   torch.empty(18, 2048, 1024,         dtype = torch.bfloat16, device = "cuda"),
            "decoder_ffn_gate_w":                 torch.empty(18, 1024, 4096,         dtype = torch.bfloat16, device = "cuda"),
            "decoder_ffn_up_w":                   torch.empty(18, 1024, 4096,         dtype = torch.bfloat16, device = "cuda"),
            "decoder_ffn_down_w":                 torch.empty(18, 4096, 1024,         dtype = torch.bfloat16, device = "cuda"),
            "decoder_action_out_proj_w":          torch.empty(1024, 32,               dtype = torch.bfloat16, device = "cuda"),
            "decoder_action_out_proj_b":          torch.empty(32,                     dtype = torch.bfloat16, device = "cuda"),
            "decoder_final_norm_mod_w":           torch.empty(1024, 3 * 1024,         dtype=torch.bfloat16, device="cuda"),
            "decoder_final_norm_mod_b":           torch.empty(3 * 1024,               dtype=torch.bfloat16, device="cuda"),
            "language_embeds":                    torch.empty(len(checkpoint['language_embeds']), 2048,  dtype = torch.bfloat16, device = "cuda"),


        }

        encoder_seq_len = num_views * 256 + self.max_prompt_len
        decoder_seq_len = chunk_size

        self.buffers = {
            'observation_images_normalized':      torch.empty(num_views, 224, 224,3,           dtype=torch.bfloat16,   device = "cuda"),
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
            'valid_encoder_len':                  torch.empty((1,),                           dtype = torch.int32, device = "cuda"),
            'encoder_logits_buf':                 torch.empty((encoder_seq_len * 8, encoder_seq_len), dtype=torch.float32,  device="cuda"),
            'encoder_attn_buf':                   torch.empty((encoder_seq_len * 8, encoder_seq_len), dtype=torch.bfloat16, device="cuda"),
            'encoder_ctx_buf':                    torch.empty((encoder_seq_len * 8, 256),     dtype=torch.bfloat16, device="cuda"),
            'decoder_rope_weights':               torch.empty(decoder_seq_len, 256,            dtype = torch.bfloat16, device = "cuda"),
            'decoder_x':                          torch.empty((decoder_seq_len, 1024),         dtype = torch.bfloat16, device = "cuda"),
            'decoder_x_buf':                      torch.empty((decoder_seq_len, 1024),         dtype=torch.bfloat16,  device = "cuda"),
            'decoder_action_buf':                 torch.empty((decoder_seq_len, 32),           dtype = torch.bfloat16, device = "cuda"),
            'decoder_time_emb':                   torch.empty((10, decoder_seq_len, 1024),         dtype = torch.bfloat16, device = "cuda"),
            'decoder_style_attn':                      torch.empty((10, 18, decoder_seq_len, 1024 * 3),         dtype = torch.bfloat16, device = "cuda"),
            'decoder_style_ffn':                      torch.empty((10, 18, decoder_seq_len, 1024 * 3),         dtype = torch.bfloat16, device = "cuda"),
            'decoder_style_final':                      torch.empty((10, decoder_seq_len, 1024 * 3),         dtype = torch.bfloat16, device = "cuda"),
            'decoder_norm_factor_buf':            torch.empty((decoder_seq_len,),              dtype = torch.bfloat16, device = "cuda"),
            'decoder_q_buf':                      torch.empty((decoder_seq_len * 8, 256),      dtype = torch.bfloat16, device = "cuda"),
            'decoder_logits_buf':                 torch.empty((decoder_seq_len * 8, encoder_seq_len + decoder_seq_len), dtype=torch.float32, device="cuda"),
            'decoder_attn_buf':                   torch.empty((decoder_seq_len * 8, encoder_seq_len + decoder_seq_len),  dtype = torch.bfloat16, device = "cuda"),
            'decoder_hidden':                     torch.empty((decoder_seq_len, 4096),         dtype = torch.bfloat16, device = "cuda"),
            'decode_split_k_buf':                 torch.empty((2, decoder_seq_len, 1024),      dtype = torch.float32, device = "cuda"),
            'x_normed_buf':                       torch.empty((decoder_seq_len, 1024),         dtype = torch.bfloat16, device = "cuda"),
            'gate_buf':                           torch.empty((decoder_seq_len, 1024),         dtype = torch.bfloat16, device = "cuda"),
        }

        prefix_alloc = self.num_views * 256 + self.max_prompt_len
        max_pos = (self.num_views * 256 + self.max_prompt_len - 1) + self.chunk_size
        position_ids = torch.arange(max_pos + 1, device="cuda")
        inv_freq = 1.0 / (10000 ** (torch.arange(0, 256, 2, dtype=torch.float32, device="cuda") / 256))
        k_phase = inv_freq[None, :] * position_ids[:, None]
        k_cos = torch.cos(k_phase).to(torch.bfloat16)
        k_sin = torch.sin(k_phase).to(torch.bfloat16)
        self._rope_table = torch.cat([k_cos[:, :, None], k_sin[:, :, None]], 2).view(-1, 256)
        self.buffers['encoder_rope_weights'].copy_(self._rope_table[:prefix_alloc])

        self.buffers['valid_encoder_len'].fill_(self.num_views * 256 + 1)
        for k, v in checkpoint.items():
            if k != "embedding_weight":
                self.weights[k].copy_(v)
        num_steps = 10
        self.weights['decoder_action_out_proj_w'] *= -1.0 / num_steps
        self.weights['decoder_action_out_proj_b'] *= -1.0 / num_steps

        for step in range(num_steps):
            matmul_1_1024_1024_bias_silu(
                self.weights['decoder_time_embeds'][step].view(1, -1),
                self.weights['decoder_time_mlp_in_w'],
                self.weights['decoder_time_mlp_in_b'],
                self.buffers['decoder_x_buf']
            )
            matmul_1_1024_1024_bias_silu(
                self.buffers['decoder_x_buf'],
                self.weights['decoder_time_mlp_out_w'],
                self.weights['decoder_time_mlp_out_b'],
                self.buffers['decoder_time_emb'][step]
            )
            for i in range(18):
                matmul_small_bias[((decoder_seq_len + 31) // 32) * (3072 // 32),](
                    self.buffers['decoder_time_emb'][step],
                    self.weights['decoder_pre_attn_norm_mod_w'][i],
                    self.buffers['decoder_style_attn'][step, i],
                    self.weights['decoder_pre_attn_norm_mod_b'][i],
                    seq_len = decoder_seq_len,
                    features = 1024,
                    hidden = 3072,
                    BLOCK_SIZE_N = 32,
                    BLOCK_SIZE_M = 32,
                    BLOCK_SIZE_K = 32
                )
                matmul_small_bias[((decoder_seq_len + 31) // 32) * (3072 // 32),](
                    self.buffers['decoder_time_emb'][step],
                    self.weights['decoder_pre_ffn_norm_mod_w'][i],
                    self.buffers['decoder_style_ffn'][step, i],
                    self.weights['decoder_pre_ffn_norm_mod_b'][i],
                    seq_len = decoder_seq_len,
                    features = 1024,
                    hidden = 3072,
                    BLOCK_SIZE_N = 32,
                    BLOCK_SIZE_M = 32,
                    BLOCK_SIZE_K = 32
                )
            matmul_small_bias[((decoder_seq_len + 31) // 32) * (3072 // 32),](
                self.buffers['decoder_time_emb'][step],
                self.weights['decoder_final_norm_mod_w'],
                self.buffers['decoder_style_final'][step],
                self.weights['decoder_final_norm_mod_b'],
                seq_len = decoder_seq_len,
                features = 1024,
                hidden = 3072,
                BLOCK_SIZE_N = 32,
                BLOCK_SIZE_M = 32,
                BLOCK_SIZE_K = 32
                )

        self.prompt_embedding = None
        self._prompt_embed_scale = None
        if self.discrete_state_input:
            if "embedding_weight" not in checkpoint:
                raise KeyError("checkpoint must contain 'embedding_weight' when discrete_state_input=True")
            emb_w = checkpoint["embedding_weight"]
            if isinstance(emb_w, np.ndarray):
                emb_w_t = torch.from_numpy(emb_w)
            else:
                emb_w_t = emb_w
            emb_w_t = emb_w_t.to(device="cuda", dtype=torch.bfloat16, non_blocking=True)
            self.prompt_embedding = nn.Embedding(
                num_embeddings=emb_w_t.shape[0],
                embedding_dim=emb_w_t.shape[1],
                device="cuda",
                dtype=torch.bfloat16,
            )
            with torch.no_grad():
                self.prompt_embedding.weight.copy_(emb_w_t)
            self._prompt_embed_scale = float(emb_w_t.shape[1] ** 0.5)
        self.encoder_seq_len = encoder_seq_len

        self.infer_graph = torch.cuda.CUDAGraph()
        self.record_infer_graph()

    def estimate_max_prompt_len(
        self,
        tokenizer: AutoTokenizer,
        task_prompt: str,
        state_dim: int,
        max_tokenize_len: int = 200,
        state_token_value: int = 255,
    ) -> int:
        task_prompt = task_prompt.strip().replace("_", " ")
        state_str = " ".join([str(int(state_token_value))] * int(state_dim))
        full_prompt = f"Task: {task_prompt}, State: {state_str};\nAction: "
        token_ids = tokenizer(
            full_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=int(max_tokenize_len),
            padding=False,
        )["input_ids"][0]
        return int(token_ids.shape[0])

    def build_prompt_embeds(
        self,
        task_prompt: str,
        state_tokens: np.ndarray
    ) -> tuple[torch.Tensor, int]:
        task_prompt = task_prompt.strip().replace("_", " ")
        state_str = " ".join(map(str, state_tokens.tolist()))
        full_prompt = f"Task: {task_prompt}, State: {state_str};\nAction: "
        token_ids = self.tokenizer(
            full_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_tokenize_len,
            padding=False,
        )["input_ids"][0].to(device="cuda", non_blocking=True)
        embeds = self.prompt_embedding(token_ids) * self._prompt_embed_scale
        return embeds, int(embeds.shape[0])

    def get_decoder_rope_weights(self, prompt_len: int) -> torch.Tensor:
        start = self.num_views * 256 + prompt_len - 1
        end = start + self.chunk_size
        return self._rope_table[start:end]

    def record_run(self):
        pi05_model(self.weights, self.buffers, self.num_views, self.encoder_seq_len)

    def record_infer_graph(self):
        for _ in range(3):
            self.record_run()
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            self.infer_graph.capture_begin()
            self.record_run()
            self.infer_graph.capture_end()

    def forward(
        self,
        observation_images_normalized: torch.Tensor,
        diffusion_noise: torch.Tensor,
        task_prompt: str = None,
        state_tokens: np.ndarray = None,
    ) -> torch.Tensor:
        if self.discrete_state_input:
            prompt_embeds, prompt_len = self.build_prompt_embeds(
                task_prompt=task_prompt,
                state_tokens=state_tokens
            )
        else:
            prompt_embeds = self.weights['language_embeds']
            prompt_len = self.weights['language_embeds'].shape[0]
        start = self.num_views * 256
        self.buffers['encoder_x'][start : start + prompt_len].copy_(prompt_embeds)
        self.buffers['valid_encoder_len'].fill_(start + prompt_len)
        self.buffers['decoder_rope_weights'].copy_(self.get_decoder_rope_weights(prompt_len))
        self.buffers['observation_images_normalized'].copy_(observation_images_normalized)
        self.buffers['diffusion_noise'].copy_(diffusion_noise)
        self.infer_graph.replay()
        return self.buffers['diffusion_noise']
