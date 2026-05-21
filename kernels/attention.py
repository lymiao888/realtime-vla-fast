import triton
import triton.language as tl


@triton.jit
def matmul_abT_scale(
    q_ptr, k_ptr, out_ptr, M : tl.constexpr, N : tl.constexpr, K : tl.constexpr,
    scale_factor: tl.constexpr,
    BLOCK_SIZE_M : tl.constexpr = 32, BLOCK_SIZE_N : tl.constexpr = 32, BLOCK_SIZE_K : tl.constexpr = 64,
):
    pid = tl.program_id(axis=0)
    psize = tl.num_programs(axis=0)
    grid_m = triton.cdiv(M, BLOCK_SIZE_M)
    grid_n = triton.cdiv(N, BLOCK_SIZE_N)

    while pid < grid_m * grid_n:
        pid_m = pid // grid_n
        pid_n = pid % grid_n
        offs_i = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_j = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for k in range(0, K, BLOCK_SIZE_K):
            offs_k = k + tl.arange(0, BLOCK_SIZE_K)
            x = tl.load(q_ptr + offs_i[:, None] * K + offs_k[None, :], mask = offs_k[None, :] < K, other = 0)
            w = tl.load(k_ptr + offs_j[:, None] * K + offs_k[None, :], mask = offs_k[None, :] < K, other = 0)
            accumulator = tl.dot(x, tl.trans(w), accumulator)
        accumulator = accumulator * scale_factor
        tl.store(out_ptr + offs_i[:, None] * N + offs_j[None, :], accumulator.to(tl.bfloat16),
                 mask = (offs_i[:, None] < M) & (offs_j[None, :] < N))
        pid += psize


@triton.jit
def softmax_kernel_mask0(
    inp_ptr, queries : tl.constexpr, keys : tl.constexpr,
    num_heads : tl.constexpr, encoder_seq_len : tl.constexpr,
    out_ptr, BLOCK_SIZE_M: tl.constexpr = 4, BLOCK_SIZE: tl.constexpr = 1024):

    pid = tl.program_id(axis=0)
    psize = tl.num_programs(axis=0)

    assert BLOCK_SIZE >= queries, f"BLOCK_SIZE must be >= N, got {BLOCK_SIZE} < {queries}"

    for i in range(pid * BLOCK_SIZE_M, queries, psize * BLOCK_SIZE_M):
        offs_i = i + tl.arange(0, BLOCK_SIZE_M)[:, None]
        offs_j = tl.arange(0, BLOCK_SIZE)[None, :]

        attn_mask = (offs_i < queries) & (offs_j < keys)
        attn_mask = attn_mask & ((offs_i >= num_heads) | (offs_j <= encoder_seq_len))
        vals = tl.load(inp_ptr + offs_i * keys + offs_j,
                       mask = attn_mask, other = -float('inf'))
        vals = tl.exp(vals - tl.max(vals, axis=1, keep_dims=True))
        vsum = tl.sum(vals, axis=1, keep_dims=True, dtype=tl.float32)
        vals = vals / vsum
        vals = vals.to(tl.bfloat16)
        tl.store(
            out_ptr + offs_i * keys + offs_j, vals,
            mask = (offs_i < queries) & (offs_j < keys)
        )


@triton.jit
def softmax_kernel_masklen(
    inp_ptr,
    queries: tl.constexpr,
    keys: tl.constexpr,
    valid_keys_len_ptr,
    out_ptr,
    BLOCK_SIZE_M: tl.constexpr = 4,
    BLOCK_SIZE: tl.constexpr = 1024,
):
    pid = tl.program_id(axis=0)
    psize = tl.num_programs(axis=0)
    big_neg = -2.3819763e38
    assert BLOCK_SIZE >= keys, f"BLOCK_SIZE must be >= keys, got {BLOCK_SIZE} < {keys}"

    valid_keys_len = tl.load(valid_keys_len_ptr).to(tl.int32)
    valid_keys_len = tl.maximum(0, tl.minimum(valid_keys_len, keys))

    for i in range(pid * BLOCK_SIZE_M, queries, psize * BLOCK_SIZE_M):
        offs_i = i + tl.arange(0, BLOCK_SIZE_M)[:, None]
        offs_j = tl.arange(0, BLOCK_SIZE)[None, :]
        attn_mask = (offs_i < queries) & (offs_j < keys) & (offs_j < valid_keys_len)
        vals = tl.load(inp_ptr + offs_i * keys + offs_j, mask=attn_mask, other=big_neg)
        vals = tl.exp(vals - tl.max(vals, axis=1, keep_dims=True))
        vsum = tl.sum(vals, axis=1, keep_dims=True, dtype=tl.float32)
        vals = vals / vsum
        tl.store(out_ptr + offs_i * keys + offs_j, vals.to(tl.bfloat16),
                 mask=(offs_i < queries) & (offs_j < keys))


@triton.jit
def softmax_kernel_prefix_suffix(
    inp_ptr,
    queries: tl.constexpr,
    keys_prefix: tl.constexpr,
    keys_suffix: tl.constexpr,
    valid_prefix_len_ptr,
    out_ptr,
    BLOCK_SIZE_M: tl.constexpr = 4,
    BLOCK_SIZE: tl.constexpr = 1024,
):
    pid = tl.program_id(axis=0)
    psize = tl.num_programs(axis=0)
    big_neg = -2.3819763e38
    total_keys: tl.constexpr = keys_prefix + keys_suffix
    assert BLOCK_SIZE >= total_keys, f"BLOCK_SIZE must be >= total_keys, got {BLOCK_SIZE} < {total_keys}"

    valid_prefix_len = tl.load(valid_prefix_len_ptr).to(tl.int32)
    valid_prefix_len = tl.maximum(0, tl.minimum(valid_prefix_len, keys_prefix))

    for i in range(pid * BLOCK_SIZE_M, queries, psize * BLOCK_SIZE_M):
        offs_i = i + tl.arange(0, BLOCK_SIZE_M)[:, None]
        offs_j = tl.arange(0, BLOCK_SIZE)[None, :]

        in_bounds = (offs_i < queries) & (offs_j < total_keys)
        is_prefix = offs_j < keys_prefix
        prefix_ok = is_prefix & (offs_j < valid_prefix_len)
        suffix_ok = (~is_prefix)
        attn_mask = in_bounds & (prefix_ok | suffix_ok)

        vals = tl.load(inp_ptr + offs_i * total_keys + offs_j, mask=attn_mask, other=big_neg)
        vals = tl.exp(vals - tl.max(vals, axis=1, keep_dims=True))
        vsum = tl.sum(vals, axis=1, keep_dims=True, dtype=tl.float32)
        vals = vals / vsum
        tl.store(out_ptr + offs_i * total_keys + offs_j, vals.to(tl.bfloat16), mask=in_bounds)
