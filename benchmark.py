import time
import argparse
import torch
import pi0_infer
import pi05_infer
from pi0_infer import Pi0Inference
from pi05_infer import Pi05Inference

def benchmark_pi0(args):
    infer = Pi0Inference({
        'language_embeds' : torch.randn(args.prompt_len, 2048, dtype = torch.bfloat16),
    }, num_views=args.num_views, chunk_size=args.chunk_size, use_cuda_graph=False)

    input_image = torch.randn(args.num_views, 224, 224, 3, dtype = torch.bfloat16).cuda()
    input_state = torch.randn(32, dtype = torch.bfloat16).cuda()
    input_noise = torch.randn(args.chunk_size, 32, dtype = torch.bfloat16).cuda()

    # Warm up
    with pi0_infer.nvtx_enabled(False):
        for _ in range(args.warmup_iterations):
            _ = infer.forward(input_image, input_state, input_noise)
            torch.cuda.synchronize()

    # Benchmark
    times = []
    for _ in range(args.iterations):
        t0 = time.time()
        _ = infer.forward(input_image, input_state, input_noise)
        torch.cuda.synchronize()
        t1 = time.time()
        times.append(t1 - t0)

    print('[Pi0 Triton]:', 'views', args.num_views, 'prompt_len', args.prompt_len, 'chunk_size', args.chunk_size)
    print('runs', len(times), 'median time per inference:', '%.3f'%(sorted(times)[len(times)//2]*1000), 'ms')

def benchmark_pi05(args):
    infer = Pi05Inference({
        'language_embeds' : torch.randn(args.prompt_len, 2048, dtype = torch.bfloat16),
    }, num_views=args.num_views, chunk_size=args.chunk_size, discrete_state_input=False, use_cuda_graph=False)

    input_image = torch.randn(args.num_views, 224, 224, 3, dtype=torch.bfloat16, device="cuda")
    input_noise = torch.randn(args.chunk_size, 32, dtype=torch.bfloat16, device="cuda")

    # Warm up
    with pi0_infer.nvtx_enabled(False):
        with pi05_infer.nvtx_enabled(False):
            for _ in range(args.warmup_iterations):
                _ = infer.forward(input_image, input_noise)
                torch.cuda.synchronize()

    # Benchmark
    times = []
    for _ in range(args.iterations):
        t0 = time.time()
        _ = infer.forward(input_image, input_noise)
        torch.cuda.synchronize()
        t1 = time.time()
        times.append(t1 - t0)

    print('[Pi05 Triton]:', 'views', args.num_views, 'chunk_size', args.chunk_size)
    print('runs', len(times), 'median time per inference:', '%.3f'%(sorted(times)[len(times)//2]*1000), 'ms')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_version", type=str, choices=["pi0", "pi05"], default="pi05")

    parser.add_argument("--num_views", type=int, default=3, help="Number of views")
    parser.add_argument("--chunk_size", type=int, default=50, help="Chunk size")
    parser.add_argument("--prompt_len", type=int, default=0, help="Pi0 prompt length")
    parser.add_argument("--warmup_iterations", type=int, default=3, help="Number of warmup iterations")
    parser.add_argument("--iterations", type=int, default=100, help="Number of benchmark iterations")

    args = parser.parse_args()

    if args.model_version == "pi0":
        benchmark_pi0(args)
    elif args.model_version == "pi05":
        benchmark_pi05(args)


if __name__ == "__main__":
    main()