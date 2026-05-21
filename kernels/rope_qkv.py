import triton
import triton.language as tl


@triton.jit
def matmul_n_2048_2560_qkv_rope(inp_ptr, weight_QKV_ptr, rope_weights_ptr, Q_ptr, K_ptr, V_ptr, seq_len : tl.constexpr, features : tl.constexpr, head_dim : tl.constexpr, num_head : tl.constexpr):
    BLOCK_SIZE_N : tl.constexpr = 64
    BLOCK_SIZE_M : tl.constexpr = 64
    BLOCK_SIZE_K : tl.constexpr = 64
    pid1 = tl.program_id(axis=0)
    psize1 = tl.num_programs(axis=0)
    pid2 = tl.program_id(axis=1)
    psize2 = tl.num_programs(axis=1)
    for i in range(pid1 * BLOCK_SIZE_N, seq_len, psize1 * BLOCK_SIZE_N):
        for j in range(pid2 * BLOCK_SIZE_M, (num_head + 2) * head_dim, psize2 * BLOCK_SIZE_M):
            acc = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_M), dtype=tl.float32)
            for k in range(0, features, BLOCK_SIZE_K):
                x = tl.load(inp_ptr + (i + tl.arange(0, BLOCK_SIZE_N)[:, None]) * features + k + tl.arange(0, BLOCK_SIZE_K), mask = tl.arange(0, BLOCK_SIZE_N)[:, None] < seq_len - i, other = 0.0)
                w = tl.load(weight_QKV_ptr + (k + tl.arange(0, BLOCK_SIZE_K)[:, None]) * ((num_head + 2) * head_dim) + j + tl.arange(0, BLOCK_SIZE_M))
                acc = tl.dot(x, w, acc)
            acc = acc.to(tl.bfloat16)
            if j < (num_head + 1) * head_dim:
                x0, x1 = tl.split(acc.reshape(BLOCK_SIZE_N, BLOCK_SIZE_M // 2, 2))
                x_cossin = tl.load(rope_weights_ptr + (i + tl.arange(0, BLOCK_SIZE_N))[:, None] * head_dim + j % head_dim + tl.arange(0, BLOCK_SIZE_M)[None, :], mask = tl.arange(0, BLOCK_SIZE_N)[:, None] < seq_len - i, other = 0.0)
                x_cos, x_sin = tl.split(x_cossin.reshape(BLOCK_SIZE_N, BLOCK_SIZE_M // 2, 2))
                x0_ = x0 * x_cos - x1 * x_sin
                x1_ = x1 * x_cos + x0 * x_sin
                acc = tl.interleave(x0_, x1_)
            if j < num_head * head_dim:
                out_ptr = Q_ptr
                out_stride = num_head * head_dim
            elif j < (num_head + 1) * head_dim:
                out_ptr = K_ptr
                out_stride = head_dim
            else:
                out_ptr = V_ptr
                out_stride = head_dim
            tl.store(out_ptr + (i + tl.arange(0, BLOCK_SIZE_N)[:, None]) * out_stride + j % out_stride + tl.arange(0, BLOCK_SIZE_M), acc, mask = tl.arange(0, BLOCK_SIZE_N)[:, None] < seq_len - i)

@triton.jit
def scaled_matmul_rope_qkv(
    inp_ptr, inp_norm_factor_ptr, seq_len: tl.constexpr, features: tl.constexpr, head_dim: tl.constexpr, num_heads: tl.constexpr,
    weight_qkv_ptr, rope_weights_ptr, q_ptr, k_ptr, v_ptr,
    BLOCK_SIZE_M : tl.constexpr = 64, BLOCK_SIZE_N : tl.constexpr = 32, BLOCK_SIZE_K : tl.constexpr = 64,
):
    pid = tl.program_id(axis=0)
    psize = tl.num_programs(axis=0)

    grid_m = triton.cdiv(seq_len, BLOCK_SIZE_M)
    grid_n = triton.cdiv((num_heads + 2) * head_dim, BLOCK_SIZE_N)

    assert head_dim % BLOCK_SIZE_N == 0, f"head_dim {head_dim} must be divisible by BLOCK_SIZE_N {BLOCK_SIZE_N}"

    while pid < grid_m * grid_n:
        pid_m = pid // grid_n
        pid_n = pid % grid_n
        start_i = pid_m * BLOCK_SIZE_M
        start_j = pid_n * BLOCK_SIZE_N
        offs_i = start_i + tl.arange(0, BLOCK_SIZE_M)[:, None]
        offs_j = start_j + tl.arange(0, BLOCK_SIZE_N)[None, :]

        norm_factor = tl.load(
            inp_norm_factor_ptr + start_i + tl.arange(0, BLOCK_SIZE_M),
            mask = start_i + tl.arange(0, BLOCK_SIZE_M) < seq_len, other=0
        )
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for k in range(0, features, BLOCK_SIZE_K):
            offs_k = k + tl.arange(0, BLOCK_SIZE_K)
            x = tl.load(
                inp_ptr + offs_i * features + offs_k[None, :],
                mask = offs_k[None, :] < features, other = 0
            )
            # x = x * norm_factor[:, None]
            w = tl.load(
                weight_qkv_ptr + offs_k[:, None] * ((num_heads + 2) * head_dim) + offs_j,
                mask = offs_k[:, None] < features, other = 0
            )
            accumulator = tl.dot(x, w, accumulator)
        accumulator = accumulator * norm_factor[:, None]

        if start_j < (num_heads + 1) * head_dim:
            x0, x1 = tl.split(accumulator.reshape(BLOCK_SIZE_M, BLOCK_SIZE_N // 2, 2))
            x_cossin = tl.load(rope_weights_ptr + offs_i * head_dim + offs_j % head_dim, mask = offs_i < seq_len, other = 0)
            x_cos, x_sin = tl.split(x_cossin.reshape(BLOCK_SIZE_M, BLOCK_SIZE_N // 2, 2))
            x0_ = x0 * x_cos - x1 * x_sin
            x1_ = x1 * x_cos + x0 * x_sin
            accumulator = tl.interleave(x0_, x1_)

        accumulator = accumulator.to(tl.bfloat16)

        if start_j < num_heads * head_dim:
            out_ptr = q_ptr
            out_stride = num_heads * head_dim
        elif start_j < (num_heads + 1) * head_dim:
            out_ptr = k_ptr
            out_stride = head_dim
        else:
            out_ptr = v_ptr
            out_stride = head_dim
        tl.store(
            out_ptr + offs_i * out_stride + offs_j % out_stride,
            accumulator,
            mask = (offs_i < seq_len) & (offs_j < (num_heads + 2) * head_dim)
        )
        pid += psize

@triton.jit
def matmul_rope_qkv(
    inp_ptr, seq_len: tl.constexpr, features: tl.constexpr, head_dim: tl.constexpr, num_heads: tl.constexpr,
    weight_qkv_ptr, rope_weights_ptr, q_ptr, k_ptr, v_ptr,
    BLOCK_SIZE_M : tl.constexpr = 64, BLOCK_SIZE_N : tl.constexpr = 32, BLOCK_SIZE_K : tl.constexpr = 64,
):
    pid = tl.program_id(axis=0)
    psize = tl.num_programs(axis=0)

    grid_m = triton.cdiv(seq_len, BLOCK_SIZE_M)
    grid_n = triton.cdiv((num_heads + 2) * head_dim, BLOCK_SIZE_N)

    assert head_dim % BLOCK_SIZE_N == 0, f"head_dim {head_dim} must be divisible by BLOCK_SIZE_N {BLOCK_SIZE_N}"

    while pid < grid_m * grid_n:
        pid_m = pid // grid_n
        pid_n = pid % grid_n
        start_i = pid_m * BLOCK_SIZE_M
        start_j = pid_n * BLOCK_SIZE_N
        offs_i = start_i + tl.arange(0, BLOCK_SIZE_M)[:, None]
        offs_j = start_j + tl.arange(0, BLOCK_SIZE_N)[None, :]

        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for k in range(0, features, BLOCK_SIZE_K):
            offs_k = k + tl.arange(0, BLOCK_SIZE_K)
            x = tl.load(
                inp_ptr + offs_i * features + offs_k[None, :],
                mask = (offs_i < seq_len) & (offs_k[None, :] < features),
                other = 0,
            )
            w = tl.load(
                weight_qkv_ptr + offs_k[:, None] * ((num_heads + 2) * head_dim) + offs_j,
                mask = (offs_k[:, None] < features) & (offs_j < (num_heads + 2) * head_dim),
                other = 0
            )
            accumulator = tl.dot(x, w, accumulator)

        if start_j < (num_heads + 1) * head_dim:
            x0, x1 = tl.split(accumulator.reshape(BLOCK_SIZE_M, BLOCK_SIZE_N // 2, 2))
            x_cossin = tl.load(rope_weights_ptr + offs_i * head_dim + offs_j % head_dim, mask = offs_i < seq_len, other = 0)
            x_cos, x_sin = tl.split(x_cossin.reshape(BLOCK_SIZE_M, BLOCK_SIZE_N // 2, 2))
            x0_ = x0 * x_cos - x1 * x_sin
            x1_ = x1 * x_cos + x0 * x_sin
            accumulator = tl.interleave(x0_, x1_)

        accumulator = accumulator.to(tl.bfloat16)

        if start_j < num_heads * head_dim:
            out_ptr = q_ptr
            out_stride = num_heads * head_dim
        elif start_j < (num_heads + 1) * head_dim:
            out_ptr = k_ptr
            out_stride = head_dim
        else:
            out_ptr = v_ptr
            out_stride = head_dim
        tl.store(
            out_ptr + offs_i * out_stride + offs_j % out_stride,
            accumulator,
            mask = (offs_i < seq_len) & (offs_j < (num_heads + 2) * head_dim)
        )
        pid += psize
