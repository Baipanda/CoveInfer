#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <random>
#include <thread>
#include <fcntl.h>
#include <unistd.h>
#include <thread>

#define CUDA_CHECK(expr)                                                                 \
    do {                                                                                 \
        cudaError_t err__ = (expr);                                                     \
        if (err__ != cudaSuccess) {                                                     \
            std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,        \
                         cudaGetErrorString(err__));                                    \
            std::exit(1);                                                               \
        }                                                                                \
    } while (0)

namespace {

struct Params {
    int iters = 100000;
    int data_size = 1 << 20;
    int nop_count = 0;
    int tamper_malicious_benign = 0;   // inserted benign malicious instructions (time impact only)
    int grid = -1;
    int block = -1;
    int repeat = 1;            // kept for compatibility; effective execution uses one kernel per run
    int verify_threads = 4096; // checksum verify subset (0 => full)
    int cpu_workers = 0;       // 0 => auto by hardware_concurrency
    uint32_t seed = 0;         // challenge seed generated on CPU side in this host process
    bool tamper_malicious_corrupt = false; // inserted malicious instruction with checksum side effect
    bool verify = false;
    bool runtime_only = false;
    bool no_copy = false;      // skip DtoH copy for runtime-only loops
};

__device__ __forceinline__ uint32_t rotl32(uint32_t x, uint32_t n) {
    return (x << n) | (x >> (32u - n));
}

__device__ __forceinline__ uint32_t ptx_mix_step(uint32_t a, uint32_t b, uint32_t salt) {
    // PTX-controlled hot-loop primitive. The sequence is intentionally verbose and
    // fixed to reduce compiler freedom and make instruction removal/reordering costly.
    asm volatile(
        "shl.b32  %0, %0, 5;        \n\t"
        "add.u32  %0, %0, %2;       \n\t"
        "mad.lo.u32 %1, %1, 1664525, 1013904223; \n\t"
        "xor.b32  %0, %0, %1;       \n\t"
        "shr.u32  %1, %1, 7;        \n\t"
        "add.u32  %0, %0, %1;       \n\t"
        "mad.lo.u32 %0, %0, 1103515245, %2; \n\t"
        : "+r"(a), "+r"(b)
        : "r"(salt));
    return a ^ rotl32(b, 11);
}

uint32_t host_mix_step(uint32_t a, uint32_t b, uint32_t salt) {
    a <<= 5;
    a += salt;
    b = b * 1664525u + 1013904223u;
    a ^= b;
    b >>= 7;
    a += b;
    a = a * 1103515245u + salt;
    return a ^ ((b << 11) | (b >> 21));
}

__global__ __launch_bounds__(256, 2) void vf_kernel(const uint32_t* __restrict__ data,
                          uint32_t* __restrict__ grid_checksum,
                          uint32_t* __restrict__ verify_out,
                          int verify_threads,
                          int iters,
                          int total_words,
                          int words_per_block,
                          int nop_count,
                          uint32_t seed,
                          int tamper_malicious_benign,
                          bool tamper_malicious_corrupt) {
    uint32_t tid_global = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t c = seed ^ (0x9e3779b9u * (tid_global + 1u));

    // 32 logical registers used by interleaved IMAD/ALU-like shift work.
    uint32_t r[32];
    #pragma unroll
    for (int i = 0; i < 32; ++i) {
        r[i] = c ^ (0x45d9f3bu * (uint32_t)(i + 1));
    }

    const uint32_t block_base = (uint32_t)blockIdx.x * (uint32_t)words_per_block;
    const uint32_t local_mask = (uint32_t)words_per_block - 1u;

    #pragma unroll 1
    for (int it = 0; it < iters; ++it) {
        // (1) D = data_ptr + (4 * C mod data_size)  <=> word index is (C mod words_per_block)
        uint32_t local_idx = c & local_mask;
        uint32_t idx = block_base + local_idx;
        if (idx >= (uint32_t)total_words) idx %= (uint32_t)total_words;

        // initiate memory read
        uint32_t d = data[idx];

        // (2) PTX-controlled mixed work: fixed verbose instruction template.
        #pragma unroll 1
        for (int s = 0; s < 32; ++s) {
            uint32_t a = r[s];
            uint32_t b = r[(s + 7) & 31];
            uint32_t salt = c ^ (uint32_t)(it * 131u + s * 17u);
            r[s] = ptx_mix_step(a, b, salt);

            // concretization to prevent algebraic shortcutting across long chains.
            if ((s & 7) == 7) {
                c ^= r[s] + (uint32_t)it;
            }
        }

        // include loaded value into checksum exactly as required
        c += d;

        // (3) Self-modification equivalent: C += C >> N, N depends on current C.
        // In CUDA we cannot safely patch instruction bytes at runtime; this is semantic equivalent.
        uint32_t N = ((c >> 27) & 31u) + 1u; // 1..32
        if (N == 32u) {
            c += 0u;
        } else {
            c += (c >> N);
        }

        // adversarial extra instructions (to test timing detectability envelope)
        #pragma unroll 1
        for (int n = 0; n < nop_count; ++n) {
            asm volatile("mov.u32 %0, %0;" : "+r"(c));
        }

        // Benign malicious-instruction injection: extra PTX work without affecting checksum value.
        uint32_t slow_acc = r[(it + 1) & 31] ^ (c + 0x9e3779b9u);
        #pragma unroll 1
        for (int s = 0; s < tamper_malicious_benign; ++s) {
            uint32_t t_idx = block_base + ((slow_acc + (uint32_t)s) & local_mask);
            if (t_idx >= (uint32_t)total_words) t_idx %= (uint32_t)total_words;
            slow_acc = ptx_mix_step(slow_acc + data[t_idx], r[(s + 13) & 31], (uint32_t)s ^ c);
        }
        asm volatile("" :: "r"(slow_acc));

        // mix register state back into checksum to bind long sequence to output.
        c ^= r[(it + 3) & 31] + rotl32(r[(it + 17) & 31], 5);
    }

    // Corrupt malicious-instruction injection: introduces side effects in checksum state.
    if (tamper_malicious_corrupt) {
        asm volatile("add.u32 %0, %0, 0x1;" : "+r"(c));
    }

    if (verify_out != nullptr && (int)tid_global < verify_threads) {
        verify_out[tid_global] = c;
    }

    // Level 1: per-warp pairwise addition (tree reduction)
    unsigned int lane = threadIdx.x & 31u;
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        c += __shfl_down_sync(0xffffffffu, c, offset);
    }

    // Level 2: per-block reduction in shared memory over warp checksums
    __shared__ uint32_t warp_sums[32];
    unsigned int warp_id = threadIdx.x >> 5;
    unsigned int warps_per_block = (blockDim.x + 31) >> 5;
    if (lane == 0) {
        warp_sums[warp_id] = c;
    }
    __syncthreads();

    uint32_t block_sum = 0;
    if (warp_id == 0) {
        block_sum = (lane < warps_per_block) ? warp_sums[lane] : 0u;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            block_sum += __shfl_down_sync(0xffffffffu, block_sum, offset);
        }

        // Level 3: grid-level aggregation in global memory with atomic add (uint32)
        if (lane == 0) {
            atomicAdd(grid_checksum, block_sum);
        }
    }
}

uint32_t cpu_thread_checksum(const std::vector<uint32_t>& data,
                             uint32_t tid_global,
                             uint32_t block_id,
                             int iters,
                             int total_words,
                             int words_per_block,
                             uint32_t seed) {
    uint32_t c = seed ^ (0x9e3779b9u * (tid_global + 1u));
    uint32_t r[32];
    for (int i = 0; i < 32; ++i) {
        r[i] = c ^ (0x45d9f3bu * (uint32_t)(i + 1));
    }

    uint32_t block_base = block_id * (uint32_t)words_per_block;
    uint32_t local_mask = (uint32_t)words_per_block - 1u;

    for (int it = 0; it < iters; ++it) {
        uint32_t local_idx = c & local_mask;
        uint32_t idx = block_base + local_idx;
        if (idx >= (uint32_t)total_words) idx %= (uint32_t)total_words;
        uint32_t d = data[idx];

        for (int s = 0; s < 32; ++s) {
            uint32_t a = r[s];
            uint32_t b = r[(s + 7) & 31];
            uint32_t salt = c ^ (uint32_t)(it * 131u + s * 17u);
            r[s] = host_mix_step(a, b, salt);

            if ((s & 7) == 7) {
                c ^= r[s] + (uint32_t)it;
            }
        }

        c += d;
        uint32_t N = ((c >> 27) & 31u) + 1u;
        if (N != 32u) c += (c >> N);
        c ^= r[(it + 3) & 31] + (((r[(it + 17) & 31] << 5) | (r[(it + 17) & 31] >> 27)));
    }

    return c;
}

bool parse_args(int argc, char** argv, Params& p) {
    for (int i = 1; i < argc; ++i) {
        auto read_int = [&](int& dst) {
            if (i + 1 >= argc) return false;
            dst = std::stoi(argv[++i]);
            return true;
        };

        if (std::strcmp(argv[i], "--iters") == 0) {
            if (!read_int(p.iters)) return false;
        } else if (std::strcmp(argv[i], "--data-size") == 0) {
            if (!read_int(p.data_size)) return false;
        } else if (std::strcmp(argv[i], "--nop-count") == 0) {
            if (!read_int(p.nop_count)) return false;
        } else if (std::strcmp(argv[i], "--tamper-malicious-benign") == 0) {
            if (!read_int(p.tamper_malicious_benign)) return false;
        } else if (std::strcmp(argv[i], "--tamper-malicious-corrupt") == 0) {
            p.tamper_malicious_corrupt = true;
        } else if (std::strcmp(argv[i], "--grid") == 0) {
            if (!read_int(p.grid)) return false;
        } else if (std::strcmp(argv[i], "--block") == 0) {
            if (!read_int(p.block)) return false;
        } else if (std::strcmp(argv[i], "--verify") == 0) {
            p.verify = true;
        } else if (std::strcmp(argv[i], "--repeat") == 0) {
            if (!read_int(p.repeat)) return false;
        } else if (std::strcmp(argv[i], "--verify-threads") == 0) {
            if (!read_int(p.verify_threads)) return false;
        } else if (std::strcmp(argv[i], "--cpu-workers") == 0) {
            if (!read_int(p.cpu_workers)) return false;
        } else if (std::strcmp(argv[i], "--no-copy") == 0) {
            p.no_copy = true;
        } else if (std::strcmp(argv[i], "--runtime-only") == 0) {
            p.runtime_only = true;
        } else if (std::strcmp(argv[i], "--help") == 0) {
            return false;
        } else {
            std::fprintf(stderr, "Unknown arg: %s\n", argv[i]);
            return false;
        }
    }

    if (p.data_size <= 0 || (p.data_size & (p.data_size - 1)) != 0) {
        std::fprintf(stderr, "data-size must be a positive power of 2\n");
        return false;
    }
    if (p.iters <= 0 || p.nop_count < 0) return false;
    return true;
}

void print_usage(const char* exe) {
    std::printf("Usage: %s [--iters N] [--data-size N] [--nop-count N] [--tamper-malicious-benign N] [--tamper-malicious-corrupt] [--grid N] [--block N] [--verify] [--verify-threads N] [--runtime-only]\n", exe);
}

} // namespace

int main(int argc, char** argv) {
    Params p;
    if (!parse_args(argc, argv, p)) {
        print_usage(argv[0]);
        return 1;
    }

    CUDA_CHECK(cudaSetDevice(0));

    {
        int fd = open("/dev/random", O_RDONLY);
        if (fd < 0) {
            std::fprintf(stderr, "Failed to open /dev/random for TRNG seed\n");
            return 1;
        }
        ssize_t n = read(fd, &p.seed, sizeof(p.seed));
        close(fd);
        if (n != (ssize_t)sizeof(p.seed)) {
            std::fprintf(stderr, "Failed to read TRNG seed from /dev/random\n");
            return 1;
        }
    }

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    if (p.block <= 0) {
        // Stable default for 40xx/50xx to keep good occupancy and register pressure.
        p.block = 256;
    }
    if (p.grid <= 0) {
        int active_blocks_per_sm = 0;
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &active_blocks_per_sm, vf_kernel, p.block, 0));

        // Use higher oversubscription to keep all SMs busy on consumer GPUs.
        int target = prop.multiProcessorCount * std::max(1, active_blocks_per_sm) * 4;
        p.grid = std::max(target, prop.multiProcessorCount * 8);
    }

    std::vector<uint32_t> host_data((size_t)p.data_size);
    for (int i = 0; i < p.data_size; ++i) {
        host_data[i] = 0x7f4a7c15u + (uint32_t)i * 0x9e3779b9u;
    }

    // per-block isolated regions for self-modification-equivalent semantics
    int words_per_block = p.data_size / std::max(1, p.grid);
    int pw2 = 1;
    while (pw2 < words_per_block) pw2 <<= 1;
    words_per_block = std::max(32, pw2 >> 1);
    words_per_block = std::min(words_per_block, p.data_size);

    uint32_t* d_data = nullptr;
    uint32_t* d_grid_checksum = nullptr;
    uint32_t* d_verify_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_data, sizeof(uint32_t) * host_data.size()));
    CUDA_CHECK(cudaMalloc(&d_grid_checksum, sizeof(uint32_t)));
    CUDA_CHECK(cudaMemcpy(d_data, host_data.data(), sizeof(uint32_t) * host_data.size(), cudaMemcpyHostToDevice));

    int verify_used = 0;
    if (p.verify) {
        const int total = p.grid * p.block;
        verify_used = (p.verify_threads <= 0) ? total : std::min(total, p.verify_threads);
        CUDA_CHECK(cudaMalloc(&d_verify_out, sizeof(uint32_t) * (size_t)verify_used));
    }

    uint32_t gpu_checksum = 0;

    auto t1 = std::chrono::high_resolution_clock::now();
    CUDA_CHECK(cudaMemset(d_grid_checksum, 0, sizeof(uint32_t)));
    vf_kernel<<<p.grid, p.block>>>(d_data, d_grid_checksum, d_verify_out, verify_used, p.iters, p.data_size, words_per_block, p.nop_count, p.seed, p.tamper_malicious_benign, p.tamper_malicious_corrupt);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(&gpu_checksum, d_grid_checksum, sizeof(uint32_t), cudaMemcpyDeviceToHost));
    auto t2 = std::chrono::high_resolution_clock::now();

    double runtime_sec = std::chrono::duration<double>(t2 - t1).count();

    bool ok = true;
    uint32_t cpu_checksum = 0;
    uint32_t gpu_checksum_verify = gpu_checksum;
    if (p.verify) {
        // CPU cost must scale with verify_used only (≤ verify_threads), not grid*block.
        // GPU writes per-thread final c into verify_out for tid < verify_used; we compare subset only.
        const int ncheck = verify_used;
        std::vector<uint32_t> h_verify((size_t)ncheck);
        CUDA_CHECK(cudaMemcpy(h_verify.data(), d_verify_out, sizeof(uint32_t) * (size_t)ncheck, cudaMemcpyDeviceToHost));

        const unsigned int hw = std::max(1u, std::thread::hardware_concurrency());
        int target_workers = (p.cpu_workers > 0) ? p.cpu_workers : (int)hw;
        const int nworkers = std::max(1, std::min(target_workers, std::max(1, ncheck)));

        std::vector<uint32_t> cpu_vals((size_t)ncheck, 0u);
        std::vector<std::thread> workers;
        workers.reserve((size_t)nworkers);

        auto worker_fn = [&](int w) {
            int begin = (ncheck * w) / nworkers;
            int end = (ncheck * (w + 1)) / nworkers;
            for (int tid = begin; tid < end; ++tid) {
                uint32_t bid = (uint32_t)(tid / p.block);
                cpu_vals[(size_t)tid] = cpu_thread_checksum(host_data, (uint32_t)tid, bid, p.iters, p.data_size,
                                                              words_per_block, p.seed);
            }
        };

        for (int w = 0; w < nworkers; ++w) {
            workers.emplace_back(worker_fn, w);
        }
        for (auto& th : workers) th.join();

        ok = true;
        for (int tid = 0; tid < ncheck; ++tid) {
            if (cpu_vals[(size_t)tid] != h_verify[(size_t)tid]) {
                ok = false;
                break;
            }
        }
        for (int tid = 0; tid < ncheck; ++tid) {
            cpu_checksum ^= cpu_vals[(size_t)tid];
        }
        gpu_checksum_verify = gpu_checksum;
    }

    if (!p.runtime_only) {
        std::printf("GPU: %s\n", prop.name);
        std::printf("Config: grid=%d block=%d iters=%d nop_count=%d repeat=%d seed=0x%08x data_size=%d words_per_block=%d\n",
                    p.grid, p.block, p.iters, p.nop_count, p.repeat, p.seed, p.data_size, words_per_block);
    }
    std::printf("Runtime: %.6f s\n", runtime_sec);
    if (!p.runtime_only) {
        if (p.verify) {
            std::printf("checksum on GPU：0x%08x\n", gpu_checksum_verify);
            std::printf("checksum on CPU：0x%08x\n", cpu_checksum);
            std::printf("verify_threads_used=%d\n", verify_used);
            std::printf("verification %s\n", ok ? "SUCCEED" : "FAILED");
        } else {
            std::printf("checksum on GPU：0x%08x\n", gpu_checksum);
        }
    }

    CUDA_CHECK(cudaFree(d_data));
    CUDA_CHECK(cudaFree(d_grid_checksum));
    if (d_verify_out) CUDA_CHECK(cudaFree(d_verify_out));

    return ok ? 0 : 2;
}
