import triton
import triton.language as tl


@triton.jit
def matmul_small_gate(inp_ptr, weight1_ptr, weight2_ptr, out_ptr, seq_len : tl.constexpr, features : tl.constexpr, hidden: tl.constexpr,
    BLOCK_SIZE_N : tl.constexpr = 128,
    BLOCK_SIZE_M : tl.constexpr = 64,
    BLOCK_SIZE_K : tl.constexpr = 32):

    pid1 = tl.program_id(axis=0)
    psize1 = tl.num_programs(axis=0)
    pid2 = tl.program_id(axis=1)
    psize2 = tl.num_programs(axis=1)

    for i in range(pid1 * BLOCK_SIZE_N, seq_len, psize1 * BLOCK_SIZE_N):
        for j in range(pid2 * BLOCK_SIZE_M, hidden, psize2 * BLOCK_SIZE_M):
            acc = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_M), dtype=tl.float32)
            acc2 = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_M), dtype=tl.float32)
            for k in range(0, features, BLOCK_SIZE_K):
                x = tl.load(
                    inp_ptr + (i + tl.arange(0, BLOCK_SIZE_N)[:, None]) * features + k + tl.arange(0, BLOCK_SIZE_K),
                    mask = i + tl.arange(0, BLOCK_SIZE_N)[:, None] < seq_len,
                    other = 0.0
                )
                w = tl.load(weight1_ptr + (k + tl.arange(0, BLOCK_SIZE_K)[:, None]) * hidden + j + tl.arange(0, BLOCK_SIZE_M))
                acc = tl.dot(x, w, acc)
                w2 = tl.load(weight2_ptr + (k + tl.arange(0, BLOCK_SIZE_K)[:, None]) * hidden + j + tl.arange(0, BLOCK_SIZE_M))
                acc2 = tl.dot(x, w2, acc2)
            acc = acc * tl.sigmoid(1.5957691216057308 * acc * (1 + 0.044715 * acc * acc))
            acc = (acc * acc2).to(tl.bfloat16)
            tl.store(out_ptr + (i + tl.arange(0, BLOCK_SIZE_N)[:, None]) * hidden + j + tl.arange(0, BLOCK_SIZE_M), acc, mask = i + tl.arange(0, BLOCK_SIZE_N)[:, None] < seq_len)

@triton.jit
def scaled_matmul_small_gate(inp_ptr, inp_norm_factor_ptr, weight1_ptr, weight2_ptr, out_ptr, seq_len : tl.constexpr, features : tl.constexpr, hidden: tl.constexpr,
    BLOCK_SIZE_N : tl.constexpr = 64,
    BLOCK_SIZE_M : tl.constexpr = 64,
    BLOCK_SIZE_K : tl.constexpr = 64):

    pid = tl.program_id(0)
    psize = tl.num_programs(0)
    grid_i = tl.cdiv(seq_len, BLOCK_SIZE_N)
    grid_j = tl.cdiv(hidden, BLOCK_SIZE_M)
    for p in range(pid, grid_i * grid_j, psize):
        i = (p // grid_j) * BLOCK_SIZE_N
        j = (p % grid_j) * BLOCK_SIZE_M
        norm_factor = tl.load(
            inp_norm_factor_ptr + i + tl.arange(0, BLOCK_SIZE_N),
            mask = i + tl.arange(0, BLOCK_SIZE_N) < seq_len, other=0
        )
        acc = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_M), dtype=tl.float32)
        acc2 = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_M), dtype=tl.float32)
        for k in range(0, features, BLOCK_SIZE_K):
            x = tl.load(
                inp_ptr + (i + tl.arange(0, BLOCK_SIZE_N)[:, None]) * features + k + tl.arange(0, BLOCK_SIZE_K),
                mask = i + tl.arange(0, BLOCK_SIZE_N)[:, None] < seq_len,
                other = 0.0
            )
            x = x * norm_factor[:, None]
            w = tl.load(weight1_ptr + (k + tl.arange(0, BLOCK_SIZE_K)[:, None]) * hidden + j + tl.arange(0, BLOCK_SIZE_M))
            acc = tl.dot(x, w, acc)
            w2 = tl.load(weight2_ptr + (k + tl.arange(0, BLOCK_SIZE_K)[:, None]) * hidden + j + tl.arange(0, BLOCK_SIZE_M))
            acc2 = tl.dot(x, w2, acc2)
        acc = acc * tl.sigmoid(1.5957691216057308 * acc * (1 + 0.044715 * acc * acc))
        acc = (acc * acc2).to(tl.bfloat16)
        tl.store(out_ptr + (i + tl.arange(0, BLOCK_SIZE_N)[:, None]) * hidden + j + tl.arange(0, BLOCK_SIZE_M), acc, mask = i + tl.arange(0, BLOCK_SIZE_N)[:, None] < seq_len)

@triton.jit
def matmul_small_res_gate(inp_ptr, weight_ptr, out_ptr, res_ptr, gate_ptr, seq_len : tl.constexpr, features : tl.constexpr, hidden : tl.constexpr,
                         BLOCK_SIZE_N : tl.constexpr, BLOCK_SIZE_M : tl.constexpr, BLOCK_SIZE_K : tl.constexpr):
    pid = tl.program_id(0)
    psize = tl.num_programs(0)
    grid_i = tl.cdiv(seq_len, BLOCK_SIZE_N)
    grid_j = tl.cdiv(hidden, BLOCK_SIZE_M)
    for p in range(pid, grid_i * grid_j, psize):
        i = (p // grid_j) * BLOCK_SIZE_N
        j = (p % grid_j) * BLOCK_SIZE_M

        acc = tl.load(
            res_ptr + (i + tl.arange(0, BLOCK_SIZE_N))[:, None] * hidden + (j + tl.arange(0, BLOCK_SIZE_M))[None, :],
            mask = ((i + tl.arange(0, BLOCK_SIZE_N))[:, None] < seq_len) & ((j + tl.arange(0, BLOCK_SIZE_M))[None, :] < hidden),
            other = 0.0
        ).to(tl.float32)
        matmul_acc = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_M), dtype=tl.float32)
        for k in range(0, features, BLOCK_SIZE_K):
            x = tl.load(
                inp_ptr + (i + tl.arange(0, BLOCK_SIZE_N))[:, None] * features + (k + tl.arange(0, BLOCK_SIZE_K))[None, :],
                mask = ((i + tl.arange(0, BLOCK_SIZE_N))[:, None] < seq_len) & ((k + tl.arange(0, BLOCK_SIZE_K))[None, :] < features),
                other = 0.0
            )
            w = tl.load(
                weight_ptr + (k + tl.arange(0, BLOCK_SIZE_K))[:, None] * hidden + (j + tl.arange(0, BLOCK_SIZE_M))[None, :],
                mask = ((k + tl.arange(0, BLOCK_SIZE_K))[:, None] < features) & ((j + tl.arange(0, BLOCK_SIZE_M))[None, :] < hidden),
                other = 0.0
            )
            matmul_acc = tl.dot(x, w, matmul_acc)

        gate = tl.load(
            gate_ptr + (i + tl.arange(0, BLOCK_SIZE_N))[:, None] * hidden + (j + tl.arange(0, BLOCK_SIZE_M))[None, :],
            mask = ((i + tl.arange(0, BLOCK_SIZE_N))[:, None] < seq_len) & ((j + tl.arange(0, BLOCK_SIZE_M))[None, :] < hidden),
            other = 0.0
        ).to(tl.float32)

        acc += matmul_acc * gate

        tl.store(
            out_ptr + (i + tl.arange(0, BLOCK_SIZE_N))[:, None] * hidden + (j + tl.arange(0, BLOCK_SIZE_M))[None, :],
            acc.to(tl.bfloat16),
            mask = ((i + tl.arange(0, BLOCK_SIZE_N))[:, None] < seq_len) & ((j + tl.arange(0, BLOCK_SIZE_M))[None, :] < hidden)
        )
