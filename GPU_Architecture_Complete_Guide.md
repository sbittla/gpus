# Complete GPU Architecture Documentation
## Detailed Technical Reference for ML Infrastructure Engineers

**Version**: 1.0  
**Target Audience**: ML Infrastructure Engineers, GPU Optimization Specialists  
**Focus**: Technical depth + Interview preparation

---

## Table of Contents

1. [Core Compute Architecture](#1-core-compute-architecture)
2. [Execution Model](#2-execution-model)
3. [Memory Hierarchy](#3-memory-hierarchy)
4. [Multi-GPU & Distributed Systems](#4-multi-gpu--distributed-systems)
5. [Profiling & Observability](#5-profiling--observability)
6. [Kubernetes & Cloud Integration](#6-kubernetes--cloud-integration)
7. [Performance Optimization](#7-performance-optimization)
8. [Concept Relationships & Mental Models](#8-concept-relationships--mental-models)
9. [Interview Preparation Guide](#9-interview-preparation-guide)

---

## 1. Core Compute Architecture

### 1.1 GPU (Graphics Processing Unit)

**Definition**: A massively parallel throughput processor optimized for executing thousands of operations simultaneously.

**Architectural Philosophy**:
- **CPU**: Few powerful cores (8-64), optimized for latency, sequential performance
- **GPU**: Many simpler cores (2,000-10,000+), optimized for throughput, parallel performance

**Core Principle**: **Latency Hiding > Latency Reduction**
- GPUs don't make individual operations faster
- They run many operations simultaneously and switch between them to hide latency

**Why GPUs Excel at ML**:
- ML workloads are dominated by matrix multiplication (Y = W × X)
- These operations are embarrassingly parallel
- Perfect match for GPU architecture

**Specifications (A100 vs H100)**:
```
Component           A100            H100
SMs                 108             132
CUDA Cores          6,912           16,896
Tensor Cores        432             528
Memory              40/80 GB        80 GB
Memory Bandwidth    2.0 TB/s        3.35 TB/s
FP16 Performance    312 TFLOPS      989 TFLOPS
NVLink Bandwidth    600 GB/s        900 GB/s
```

**Interview Soundbite**: "GPUs trade single-thread latency for extreme parallel throughput, achieving performance through massive parallelism and latency hiding rather than making individual operations faster."

---

### 1.2 Streaming Multiprocessor (SM)

**Definition**: The fundamental execution unit inside a GPU, equivalent to a CPU core but designed for parallel execution.

**Internal Architecture**:
- **Execution Units**: CUDA cores (64-128), Tensor cores (4-8), Special Function Units
- **Scheduling**: 4 warp schedulers, dispatch units
- **Memory**: Register file (64-256KB), Shared memory/L1 cache (64-192KB configurable)

**How SMs Work**:
1. Thread blocks assigned to SMs
2. SM divides blocks into warps (32 threads each)
3. Warp scheduler selects ready warps for execution
4. When one warp stalls (memory access), scheduler instantly switches to another
5. This continues until all warps complete

**Latency Hiding Example**:
```
Cycle 0-10:   Warp 0 executing math
Cycle 11:     Warp 0 requests global memory (400 cycle latency!)
Cycle 12-411: Warps 1,2,3... execute while Warp 0 waits
Cycle 412:    Warp 0 data arrives, continues execution
```

**Why SM Utilization Matters**:
- You pay for 100% of GPU
- Idle SMs = wasted money
- A100 with 108 SMs at 50% utilization = ~$1.50/hour wasted on cloud

**Optimization Strategy**:
- Increase grid dimensions (more blocks)
- Balance block size (128-512 threads typically optimal)
- Avoid small kernels (fuse operations)
- Monitor with nvidia-smi and profile with Nsight

---

### 1.3 CUDA Core

**Definition**: Scalar arithmetic logic unit (ALU) capable of one FP32/FP64/INT operation per clock cycle.

**Capabilities**:
- FP32 (single precision)
- FP64 (double precision, often 1/2 or 1/32 rate)
- INT32 (integer operations)
- Logical operations

**Use Cases**:
- Element-wise operations (ReLU, sigmoid, tanh)
- Preprocessing (normalization, data augmentation)
- Control flow logic
- Non-matrix computations

**Performance Gap**:
- Matrix multiply on CUDA cores (A100): ~19 TFLOPS
- Matrix multiply on Tensor cores (A100): ~312 TFLOPS
- **16x slower for ML workloads!**

**Critical Insight**: If your ML workload primarily uses CUDA cores instead of Tensor cores, you're missing 10-50x performance gains.

**Verification**:
```bash
ncu --metrics sm__inst_executed_pipe_tensor.avg.pct_of_peak_sustained_active

# Low percentage = missing tensor acceleration
```

---

### 1.4 Tensor Core

**Definition**: Specialized hardware accelerator for mixed-precision matrix multiply-accumulate operations (D = A × B + C).

**Why They Exist**:
- 90%+ of deep learning compute is matrix multiplication
- Traditional CUDA cores: perform n×k×m individual operations serially
- Tensor cores: perform 4×4 or 8×8 matrix chunks simultaneously
- **Massive parallelism boost**: 64x for 1024×1024 matrix

**Supported Data Types**:

| Format | Precision | Use Case | A100 Performance |
|--------|-----------|----------|------------------|
| FP16 | 16-bit float | Training, inference | 312 TFLOPS |
| BF16 | 16-bit brain float | Training (better range) | 312 TFLOPS |
| TF32 | 19-bit tensor float | Drop-in FP32 replacement | 156 TFLOPS |
| INT8 | 8-bit integer | Inference only | 624 TOPS |
| FP8 | 8-bit float (H100+) | Training & inference | 1979 TFLOPS |

**Enabling Tensor Cores** (PyTorch):
```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for data, target in dataloader:
    with autocast():  # Automatically uses FP16 for matmuls
        output = model(data)
        loss = criterion(output, target)
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

**Optimization Strategies**:
1. **Dimension Alignment**: Use multiples of 8 (preferably 16)
   ```python
   # Bad: batch_size=37, hidden_dim=511
   # Good: batch_size=40, hidden_dim=512
   ```

2. **Mixed Precision**: Always train with FP16/BF16

3. **Framework Support**: Use cuDNN, cuBLAS, TensorRT

4. **Batch Sizing**: Larger batches = better Tensor core utilization

**Real-World Impact**: BERT training FP32 → FP16: 300 → 2400 samples/sec (**8x speedup**)

---

### 1.5 Warp

**Definition**: Group of 32 threads that execute in lockstep using SIMT (Single Instruction, Multiple Thread) model.

**SIMT Execution**:
- All 32 threads execute same instruction simultaneously
- But on different data
- Like 32 students following same instruction, but on their own numbers

**Warp Divergence Problem**:

When threads in a warp take different code paths, GPU must serialize execution:

```cuda
// DIVERGENT CODE (BAD)
if (threadIdx.x % 2 == 0) {
    data[tid] = tid * 2;  // 16 threads execute
} else {
    data[tid] = tid * 3;  // 16 threads execute
}

// Actual execution:
// Step 1: Active mask = even threads, execute path A, odd threads idle
// Step 2: Active mask = odd threads, execute path B, even threads idle
// Result: 50% efficiency (2x slower)
```

**Worst Case**:
```cuda
if (threadIdx.x == 0) {
    // Only 1 thread active, 31 idle
    // 3% efficiency!
}
```

**Avoiding Divergence**:

```cuda
// BAD: Divergence within warp
if (threadIdx.x < threshold) {
    // Some threads
} else {
    // Other threads
}

// GOOD: Entire warps take same path
if (threadIdx.x / 32 < threshold / 32) {
    // All threads in warp
}

// BETTER: Branchless
int multiplier = (threadIdx.x % 2 == 0) ? 2 : 3;
data[threadIdx.x] = threadIdx.x * multiplier;
```

**Measuring Divergence**:
```bash
ncu --metrics smsp__sass_average_branch_targets_threads_uniform.pct
# Target: >95%
```

**Performance Impact**: 50-80% loss with typical divergence

---

### 1.6 Thread Block

**Definition**: Group of threads (128-1024) scheduled together on single SM, can cooperate through shared memory and synchronization.

**Key Properties**:
- Share same shared memory pool
- Can synchronize via `__syncthreads()`
- Stay on same SM until completion
- Independent of other blocks

**Organization Examples**:
```cuda
// 1D Block
dim3 block(256);
dim3 grid(100);
// Total threads: 25,600

// 2D Block (image processing)
dim3 block(16, 16);  // 256 threads
dim3 grid(64, 64);   // 4,096 blocks

// 3D Block (volumetric data)
dim3 block(8, 8, 4);  // 256 threads
dim3 grid(32, 32, 16);
```

**Why Blocks Matter**: Enable cooperation through shared memory

```cuda
__global__ void sum_reduction(float* input, float* output) {
    __shared__ float cache[256];  // Shared by all threads in block
    
    int tid = threadIdx.x;
    cache[tid] = input[blockIdx.x * blockDim.x + tid];
    __syncthreads();  // Wait for all threads
    
    // Tree reduction within block
    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (tid < s) cache[tid] += cache[tid + s];
        __syncthreads();
    }
    
    if (tid == 0) output[blockIdx.x] = cache[0];
}
```

**Sizing Guidelines**:

| Block Size | Pros | Cons | When to Use |
|------------|------|------|-------------|
| 64 threads | Low resource usage | Poor SM utilization | High register/shared mem usage |
| 128-512 | Balanced | - | Most workloads (sweet spot) |
| 1024 | Max work/block | May reduce occupancy | Minimal resource usage |

**Resource Balancing** (A100):
- Max threads/SM: 2048
- Max blocks/SM: 32
- Shared memory/SM: 164KB

Choose block size to maximize threads while staying within resource limits.

---

## 2. Execution Model

### 2.1 Kernel

**Definition**: GPU-side function that executes in parallel across N threads.

**Syntax**:
```cuda
__global__ void kernelName(parameters) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // Work here
}

// Launch from CPU
kernelName<<<gridDim, blockDim, sharedMem, stream>>>(args);
```

**Lifecycle**:
1. **Launch** (CPU): Kernel enqueued, CPU continues immediately (async)
2. **Schedule** (GPU): Blocks assigned to SMs
3. **Execute** (SM): Warps execute instructions
4. **Complete**: Results written to memory

**Launch Overhead**: ~5-10 microseconds per kernel

**Performance Consideration**: Too many small kernels kills throughput

```python
# BAD: 1000 kernel launches (10ms overhead)
for i in range(1000):
    tiny_kernel<<<1, 32>>>(data[i])

# GOOD: 1 kernel launch (10μs overhead, 1000x faster!)
big_kernel<<<1000, 32>>>(data)
```

**Kernel Fusion**: Combining multiple operations into one kernel

```cuda
// Before: 3 separate kernels
relu_kernel<<<n, 256>>>(x);
batchnorm_kernel<<<n, 256>>>(x);
dropout_kernel<<<n, 256>>>(x);

// After: 1 fused kernel
__global__ void fused_kernel(float* x, ...) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    x[i] = relu(x[i]);
    x[i] = batchnorm(x[i], ...);
    x[i] = dropout(x[i], ...);
}
```

**Benefits of Fusion**:
- 3x less launch overhead
- No global memory round-trips between ops
- Better instruction cache usage

---

### 2.2 Occupancy

**Definition**: Ratio of active warps to maximum possible warps on SM

```
Occupancy = (Active Warps per SM) / (Maximum Warps per SM)

Example (A100):
- Max warps per SM: 64
- Your kernel: 48 active warps
- Occupancy: 75%
```

**Why It Matters**: More warps = more latency hiding opportunities

**Low Occupancy Scenario** (25%):
```
Only 16 warps active
Warp 0 stalls → switch to Warp 1
Warp 1 stalls → switch to Warp 2
...
Warp 15 stalls → NO MORE WARPS!
SM goes idle (wasted cycles)
```

**High Occupancy Scenario** (75%):
```
48 warps active
Always has ready warp to execute
SM never idle
```

**Factors Limiting Occupancy**:

1. **Register Usage**:
   ```
   SM register file: 65,536 registers (A100)
   Max warps: 64
   
   If kernel uses 64 registers/thread:
   Max warps: 65,536 / (64 * 32) = 32
   Occupancy: 50%
   ```

2. **Shared Memory**:
   ```
   SM shared memory: 164 KB (A100)
   
   If kernel uses 16 KB shared mem/block:
   Max blocks: 164 / 16 = 10
   (Limited by resource, not hardware max of 32)
   ```

3. **Block Size**:
   ```
   Max threads/SM: 2048
   Block size: 96 threads (not multiple of 32!)
   
   Max blocks: 2048 / 96 = 21
   Wasted: 1 warp worth of resources
   ```

**The Sweet Spot**: 60-80% occupancy often optimal

**Why Not 100%?**:
- Trade-off: More resources per thread = better algorithm
- Example: 75% occupancy with optimized algorithm beats 100% with naive algorithm

**Measuring Occupancy**:
```bash
# Compile-time estimate
nvcc --ptxas-options=-v kernel.cu
# Output: Theoretical occupancy: 75%

# Runtime measurement
ncu --metrics sm__warps_active.avg.pct_of_peak_sustained_active ./app
```

---

## 3. Memory Hierarchy

### Memory Overview

**The Fundamental Problem**: Compute is fast, memory is slow

```
A100 Compute: 312 TFLOPS (312 × 10^12 ops/sec)
A100 Memory BW: 2 TB/s (2 × 10^12 bytes/sec)

To saturate compute: Need 156 ops per byte!
Most kernels: 1-10 ops per byte → memory-bound
```

**Memory Pyramid** (fastest → slowest):
```
Registers:       1 cycle,    256KB/SM,   thread-private
Shared Memory:   ~20 cycles, 164KB/SM,   block-shared
L2 Cache:        ~200 cycles, 40MB,      GPU-wide, automatic
Global Memory:   ~400 cycles, 40-80GB,   all threads
```

---

### 3.1 Registers

**Fastest Memory**: 1-cycle latency, private to each thread

**Allocation** (per-thread basis):
```cuda
float x = 0.0f;        // 1 register
float y = 1.0f;        // 1 register
int i = threadIdx.x;   // 1 register
```

**Register Budget** (A100):
```
Total/SM: 65,536 registers
Max threads/SM: 2,048
Max per thread: 32 registers

If kernel uses 64 registers/thread:
Max threads: 1,024 (50% occupancy)
```

**Register Spilling**: When kernel uses too many variables, some "spill" to memory

```bash
nvcc --ptxas-options=-v kernel.cu
# Output: Used 80 registers, 128 bytes lmem
#                              ^^^^ SPILLING!

# Performance impact: 400x slower!
# Register access: 1 cycle
# Spilled access: 400+ cycles
```

**Optimization**:
```cuda
// Before: Many variables
float temp1 = a + b;
float temp2 = c + d;
// ... temp3-temp10

// After: Reuse temporaries
float temp = a + b;
result1 = temp;
temp = c + d;
result2 = temp;
```

---

### 3.2 Shared Memory

**On-Chip Scratchpad**: ~20 cycle latency, shared by thread block

**Why It Exists**: 20x faster than global memory, enables data reuse

```cuda
// Global memory (slow): 400 cycles × 100 accesses = 40,000 cycles
for (int i = 0; i < 100; i++) {
    float x = global_data[index];
}

// Shared memory (fast): 400 + (20 × 100) = 2,400 cycles (16x faster!)
__shared__ float shared_data[256];
shared_data[tid] = global_data[index];  // Load once
__syncthreads();

for (int i = 0; i < 100; i++) {
    float x = shared_data[tid];  // Reuse
}
```

**Specification** (A100):
- Size: 164 KB per SM (configurable split with L1)
- Lifetime: Duration of block
- Access: All threads in block

**Classic Use Case: Matrix Multiplication Tiling**

```cuda
__global__ void matmul_tiled(float* A, float* B, float* C, int N) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    float sum = 0.0f;
    
    for (int t = 0; t < N / TILE_SIZE; t++) {
        // Load tile to shared memory
        As[threadIdx.y][threadIdx.x] = A[row * N + t * TILE_SIZE + threadIdx.x];
        Bs[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * N + col];
        __syncthreads();
        
        // Compute using shared memory
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        __syncthreads();
    }
    
    C[row * N + col] = sum;
}
```

**Speedup**: 10-20x faster than naive implementation

**Bank Conflicts**:

Shared memory organized into 32 banks. Multiple threads accessing same bank (different addresses) = serialization.

```
Bank 0: Addresses 0, 32, 64, ...
Bank 1: Addresses 1, 33, 65, ...
...
Bank 31: Addresses 31, 63, 95, ...

// No conflict (sequential)
float x = data[threadIdx.x];

// Conflict (stride = 32)
float x = data[threadIdx.x * 32];  // All threads → Bank 0! (32x slower)
```

**Solution: Padding**
```cuda
// Bad: 32×32 array
__shared__ float bad[32][32];

// Good: Add padding column
__shared__ float good[32][33];
```

---

### 3.3 L2 Cache

**Hardware-Managed Cache**: ~200 cycle latency, shared across all SMs

**Specification** (A100):
- Size: 40 MB
- Automatic management (no programmer control)
- Caches reads and writes

**How It Works**:
```cuda
// First access: L2 miss → 400 cycles (fetch from global)
float x = global_array[i];

// Second access (if still in L2): L2 hit → 200 cycles
float y = global_array[i];

// Much later (evicted): L2 miss → 400 cycles again
```

**Cache Line**: 128 bytes (32 floats)
- Accessing 1 float loads entire cache line
- Spatial locality automatically exploited

**Optimization for L2**:
1. **Keep working sets small** (fit in 40MB)
2. **Sequential access** (cache lines useful)
3. **Reuse data** (temporal locality)

**L2 Hit Rate Measurement**:
```bash
ncu --metrics lts__t_sector_hit_rate.pct
# Target: >70%
```

---

### 3.4 Global Memory (HBM)

**Main GPU DRAM**: Largest but slowest memory tier

**Specification** (A100):
- Capacity: 40-80 GB
- Bandwidth: 1.9-2.0 TB/s
- Latency: ~400-800 cycles

**Bandwidth vs Latency**:
- **Bandwidth**: Total data transfer rate (TB/s)
- **Latency**: Time for single access (cycles)
- High bandwidth doesn't mean low latency!
- GPU hides latency through parallelism

**Memory Allocation**:
```cpp
// Device memory
float* d_data;
cudaMalloc(&d_data, size);
cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice);

// Kernel use
kernel<<<grid, block>>>(d_data);

// Copy back
cudaMemcpy(h_result, d_data, size, cudaMemcpyDeviceToHost);
cudaFree(d_data);
```

**Unified Memory** (easier but slower):
```cpp
float* data;
cudaMallocManaged(&data, size);  // Accessible from CPU & GPU
data[0] = 1.0f;  // CPU
kernel<<<grid, block>>>(data);  // GPU
```

**Pinned Memory** (faster transfers):
```cpp
float* h_data;
cudaMallocHost(&h_data, size);  // Page-locked
cudaMemcpy(d_data, h_data, size, H2D);  // 2-3x faster
cudaFreeHost(h_data);
```

**Optimization Strategies**:
1. **Minimize transfers** (keep data on GPU)
2. **Async transfers** (overlap with compute)
3. **Prefetching** (unified memory)
4. **Coalesce accesses** (combine warp requests)

---

### 3.5 Memory Coalescing

**Definition**: Combining multiple thread memory requests into minimal transactions

**Problem**: Memory transaction unit = 128 bytes (32 consecutive floats)

**Without Coalescing**:
```
32 threads request 32 random locations
→ 32 separate transactions
→ Fetch 32 × 128 = 4,096 bytes
→ Use only 32 × 4 = 128 bytes
→ Efficiency: 3%
```

**With Coalescing**:
```
32 threads request sequential locations
→ 1 combined transaction
→ Fetch 1 × 128 = 128 bytes
→ Use all 128 bytes
→ Efficiency: 100%
→ 32x faster!
```

**Perfect Pattern**:
```cuda
int i = blockIdx.x * blockDim.x + threadIdx.x;
float x = data[i];

// Thread 0 → data[0]
// Thread 1 → data[1]
// ...
// Thread 31 → data[31]
// All in 1 cache line → 1 transaction
```

**Violations**:

```cuda
// BAD: Stride of 2 (50% efficiency, 2 transactions)
int i = (blockIdx.x * blockDim.x + threadIdx.x) * 2;

// TERRIBLE: Stride of 1024 (3% efficiency, 32 transactions)
int i = threadIdx.x * 1024;

// WORST: Random (3% efficiency, likely 32 transactions)
int i = random_indices[threadIdx.x];
```

**Struct of Arrays (SoA) vs Array of Structs (AoS)**:

```cuda
// AoS (BAD for GPU)
struct Particle {
    float x, y, z;
    float vx, vy, vz;
};
Particle* particles;

// Thread 0: loads particles[0] (bytes 0-23)
// Thread 1: loads particles[1] (bytes 24-47)
// 6 particles per cache line → 6 transactions for 32 threads
// Wastes y, z, vy, vz data

// SoA (GOOD for GPU)
struct ParticlesSoA {
    float* x;   // All x values contiguous
    float* vx;  // All vx values contiguous
};

// Thread 0-31: loads x[0-31] → 1 transaction! Perfect coalescing
```

**Speedup**: 5-10x improvement with SoA

**Measuring Coalescing**:
```bash
ncu --metrics l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum              l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum

# Efficiency = bytes / (requests × 128)
# Target: >80%
```

---

## 4. Multi-GPU & Distributed Systems

### 4.1 NVLink

**Definition**: NVIDIA's high-speed interconnect for direct GPU-to-GPU communication

**Bandwidth Comparison**:
```
PCIe 4.0 x16:  ~32 GB/s bidirectional
PCIe 5.0 x16:  ~64 GB/s bidirectional
NVLink 3.0:    600 GB/s bidirectional (A100)
NVLink 4.0:    900 GB/s bidirectional (H100)

Speedup: 18-28x faster than PCIe!
```

**Use Cases**:
1. **Data Parallelism**: Gradient synchronization across GPUs
2. **Tensor Parallelism**: Split model across GPUs (attention heads, layers)
3. **Pipeline Parallelism**: Model stages on different GPUs
4. **Large Models**: Models exceeding single GPU memory

**Performance Impact** (8x A100, ResNet-50):
```
PCIe only: 18,000 images/sec
With NVLink: 52,000 images/sec
Speedup: 2.9x
```

**Scaling Efficiency**:
```
1 GPU:   100% (baseline)
2 GPUs:  195% (97.5% efficiency) ← NVLink
4 GPUs:  385% (96% efficiency)
8 GPUs:  750% (94% efficiency)

vs PCIe:
8 GPUs:  520% (65% efficiency) ← PCIe bottleneck
```

**Verification**:
```bash
nvidia-smi topo -m

# Output shows connectivity:
#      GPU0  GPU1  GPU2  GPU3
# GPU0  X    NV12  NV12  NV12
# GPU1 NV12   X    NV12  NV12
# NV12 = 12 NVLink connections

# Monitor utilization
nvidia-smi nvlink --status
```

---

### 4.2 InfiniBand

**Definition**: High-performance networking for GPU clusters (multi-node communication)

**Specification**:
- **HDR**: 200 Gb/s (25 GB/s) per port
- **NDR**: 400 Gb/s (50 GB/s) per port
- **Latency**: ~1-2 microseconds
- **RDMA**: Bypass CPU for direct memory access

**Why InfiniBand for ML**:

Ethernet TCP/IP:
- Latency: 50-100 microseconds
- CPU overhead: 30-50%

InfiniBand RDMA:
- Latency: 1-2 microseconds
- CPU overhead: <5%

**Multi-Node Architecture**:
```
Node 0: [8x A100] ←NVLink→ [InfiniBand HCA]
                             ↓
Node 1: [8x A100] ←NVLink→ [InfiniBand HCA]
                             ↓
        InfiniBand Switch (200 Gb/s)
                             ↓
Node 2-3: Similar topology

Intra-node: NVLink (600 GB/s)
Inter-node: InfiniBand (200 Gb/s)
```

**Communication Pattern** (All-Reduce for gradients):
```
32 GPUs across 4 nodes, 1GB gradients

Intra-node (NVLink): 1.6 ms
Inter-node (IB):     40 ms
Total: ~42 ms

Without InfiniBand: 500+ ms (10x slower)
```

**Insight**: Most distributed training is network-bound, not compute-bound!

**Optimization**:
1. **Gradient Compression**: FP32 → FP16 (2x less data)
2. **Gradient Accumulation**: Fewer sync points
3. **Overlapping**: Start sync during backward pass

**Monitoring**:
```bash
# Check link
ibstat
# Output: State: Active, Rate: 200 Gb/s

# Test bandwidth
ib_write_bw -d mlx5_0

# NCCL tests
./all_reduce_perf -b 8 -e 1G -f 2 -g 8
```

---

## 5. Profiling & Observability

### 5.1 nvidia-smi

**Real-time Monitoring**:
```bash
nvidia-smi -l 1  # Update every 1 second

# Key metrics:
# - GPU-Util: % time GPU busy (target: >80%)
# - Memory-Usage: Allocated memory
# - Power: Current draw vs max
# - Temp: Temperature (throttles at 83-90°C)
```

**Programmatic Access**:
```python
import subprocess
result = subprocess.run(
    ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used',
     '--format=csv,noheader,nounits'],
    capture_output=True, text=True
)
gpu_util, mem_used = result.stdout.strip().split(', ')
```

---

### 5.2 Nsight Systems

**Purpose**: System-wide timeline profiling

**Usage**:
```bash
nsys profile -o profile python train.py
nsys-ui profile.nsys-rep  # Open GUI
```

**What It Shows**:
- Kernel execution timeline
- CPU activity
- Memory transfers (H2D, D2H)
- CUDA API calls
- Thread synchronization

**Key Insights**:
- Gaps = idle time (optimize!)
- Overlapping = good parallelism
- Sequential = potential for pipelining

**NVTX Annotations** (custom markers):
```python
import torch.cuda.nvtx as nvtx

nvtx.range_push("forward")
output = model(data)
nvtx.range_pop()

nvtx.range_push("backward")
loss.backward()
nvtx.range_pop()
```

---

### 5.3 Nsight Compute

**Purpose**: Deep kernel-level analysis

**Usage**:
```bash
ncu -o profile ./app
ncu-ui profile.ncu-rep  # Open GUI

# Specific metrics
ncu --metrics sm__throughput.avg.pct_of_peak_sustained_active ./app
```

**Key Metrics**:

1. **SM Utilization**:
   ```bash
   ncu --metrics sm__throughput.avg.pct_of_peak_sustained_active
   # >80% = good (compute-bound)
   # <50% = likely memory-bound
   ```

2. **Memory Bandwidth**:
   ```bash
   ncu --metrics dram__throughput.avg.pct_of_peak_sustained_active
   # >70% = good utilization
   # <50% = poor access patterns
   ```

3. **Tensor Core Usage**:
   ```bash
   ncu --metrics sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active
   # Should be >70% for ML workloads
   ```

4. **Occupancy**:
   ```bash
   ncu --metrics sm__warps_active.avg.pct_of_peak_sustained_active
   # Target: 60-80%
   ```

**Roofline Analysis**:
```bash
ncu --set roofline ./app

# Shows:
# - Below memory bandwidth line = memory-bound
# - Below compute line = compute-bound
# - Optimization direction clear
```

---

### 5.4 Prometheus + Grafana

**Architecture**:
```
GPUs → DCGM Exporter → Prometheus → Grafana
                           ↓
                    Alerting + Storage
```

**DCGM Exporter Setup**:
```bash
docker run -d --gpus all -p 9400:9400   nvcr.io/nvidia/k8s/dcgm-exporter:latest

curl localhost:9400/metrics | grep DCGM
```

**Key Metrics**:
```
DCGM_FI_DEV_GPU_UTIL              # GPU utilization
DCGM_FI_DEV_FB_USED               # Memory used
DCGM_FI_PROF_PIPE_TENSOR_ACTIVE   # Tensor core activity
DCGM_FI_DEV_GPU_TEMP              # Temperature
DCGM_FI_DEV_POWER_USAGE           # Power draw
```

**Alerting Rules**:
```yaml
- alert: LowGPUUtilization
  expr: DCGM_FI_DEV_GPU_UTIL < 20
  for: 5m

- alert: LowTensorCoreUsage
  expr: DCGM_FI_PROF_PIPE_TENSOR_ACTIVE < 30
  for: 10m
```

---

## 6. Kubernetes & Cloud Integration

### 6.1 GPU Device Plugin

**Installation**:
```bash
kubectl create -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/main/nvidia-device-plugin.yml

# Verify
kubectl get nodes -o yaml | grep nvidia.com/gpu
```

**Pod Request**:
```yaml
resources:
  limits:
    nvidia.com/gpu: 1  # Request 1 GPU
```

---

### 6.2 MIG (Multi-Instance GPU)

**Definition**: Partition single A100/H100 into up to 7 isolated GPU instances

**Why**: Better utilization for small workloads

**Example**:
```
1 A100 running 1 small inference job: 15% utilization (waste!)

With MIG:
1 A100 → 7x MIG 1g.5gb instances
7 small jobs running: 95% utilization!
```

**MIG Profiles** (A100 40GB):
```
MIG 1g.5gb:  1/7 GPU, 14 SMs, 5GB  (7 instances possible)
MIG 2g.10gb: 2/7 GPU, 28 SMs, 10GB (3 instances possible)
MIG 3g.20gb: 3/7 GPU, 42 SMs, 20GB (2 instances possible)
```

**Enable MIG**:
```bash
# Enable mode
sudo nvidia-smi -i 0 -mig 1

# Create instances
sudo nvidia-smi mig -cgi 1g.5gb,1g.5gb,1g.5gb,1g.5gb,1g.5gb,1g.5gb,1g.5gb

# Create compute instances
sudo nvidia-smi mig -cci
```

**Kubernetes Usage**:
```yaml
resources:
  limits:
    nvidia.com/mig-1g.5gb: 1
```

**Use Cases**:
- ✅ Inference (small models)
- ✅ Development/testing
- ✅ Multi-tenant clusters
- ❌ Large model training
- ❌ Workloads needing NVLink

---

### 6.3 Cost Optimization

**GPU Utilization Tracking**:
```promql
# Wasted GPU hours (util < 50%)
sum(
  (DCGM_FI_DEV_GPU_UTIL < 50) * on(gpu) group_left() node_info
) / 3600

# Cost calculation (example: $3/GPU/hr)
(sum(DCGM_FI_DEV_GPU_UTIL < 50) / 3600) * 3
```

**Right-Sizing**:
```python
for pod in pods:
    if get_gpu_util(pod) < 0.3:
        print(f"Pod {pod}: Use MIG instead")
        savings = calculate_savings("1g.5gb")
        print(f"Save ${savings}/month")
```

**Spot Instances**:
```
On-demand p4d.24xlarge: $32.77/hr (8x A100)
Spot p4d.24xlarge:      ~$10/hr (70% savings!)
```

---

## 7. Performance Optimization

### 7.1 Common Bottlenecks & Fixes

#### Low GPU Utilization

**Causes**:
1. **Small batch size** → Increase batch size
2. **CPU bottleneck** → GPU-accelerated data pipeline (DALI)
3. **I/O bottleneck** → Prefetch data, more workers

```python
# Fix I/O
dataloader = DataLoader(
    dataset,
    batch_size=64,
    num_workers=8,
    pin_memory=True,
    prefetch_factor=2
)
```

#### Memory-Bound Kernels

**Diagnosis**:
- High memory BW (>70%)
- Low SM utilization (<50%)

**Fixes**:
1. **Tiling** (use shared memory)
2. **Kernel fusion** (reduce memory passes)
3. **Increase arithmetic intensity**

#### PCIe Bottlenecks

**Fixes**:
1. **Keep data on GPU** (minimize transfers)
2. **Async transfers** (overlap with compute)
3. **Pinned memory** (faster transfers)

```python
# Bad: Transfer loss to CPU every iteration
loss_cpu = loss.item()  # D2H transfer!

# Good: Accumulate on GPU, transfer once
losses.append(loss)
final_loss = torch.stack(losses).mean().item()
```

#### Kernel Launch Overhead

**Fix**: Kernel fusion

```python
# Enable PyTorch JIT fusion
with torch.jit.fuser("fuser2"):
    for i in range(1000):
        x = x + 1
        x = x * 2
        x = torch.relu(x)
# Thousands of kernels → ~10 fused kernels
```

#### Warp Divergence

**Fix**: Branchless code

```cuda
// Bad: Divergent
if (threadIdx.x % 2 == 0) {
    data[i] = x * 2;
} else {
    data[i] = x * 3;
}

// Good: Branchless
int mult = (threadIdx.x % 2 == 0) ? 2 : 3;
data[i] = x * mult;
```

---

## 8. Concept Relationships & Mental Models

### Hierarchical Structure

```
GPU
├── SMs (108 in A100)
│   ├── CUDA Cores (64-128)
│   ├── Tensor Cores (4-8)
│   ├── Warp Schedulers (4)
│   ├── Registers (64-256KB)
│   └── Shared Memory (164KB)
├── L2 Cache (40MB)
└── Global Memory (40-80GB)
```

### Execution Flow

```
1. CPU launches kernel<<<grid, block>>>()
2. GPU assigns blocks to SMs
3. SM divides blocks into warps (32 threads each)
4. Warp scheduler executes instructions
5. Memory hierarchy accessed (Registers → Shared → L2 → Global)
6. Results written back
```

### Memory Flow

```
Thread → Registers (1 cycle)
      → Shared Memory (20 cycles)
      → L2 Cache (200 cycles)
      → Global Memory (400 cycles)
      → PCIe/NVLink → CPU/Other GPUs
      → InfiniBand → Remote Nodes
```

### Core Principles

**1. Latency Hiding Through Parallelism**
```
Single thread: 400 cycle latency
10,000 threads: Effective ~0 latency (hidden by parallelism)

"GPU doesn't make operations faster, it hides the wait"
```

**2. Memory is the Bottleneck**
```
Compute: 312 TFLOPS
Memory: 2 TB/s

Need 156 ops/byte to saturate compute
Most kernels: 1-10 ops/byte → memory-bound

"Optimize memory first, compute second"
```

**3. Locality is Everything**
```
Register: 1 cycle
Global: 400 cycles

400x difference!

"A byte saved is 400 cycles earned"
```

**4. Granularity Matters**
```
Too fine: Overhead dominates
Too coarse: Idle resources

Sweet spot: 128-512 thread blocks, fused kernels

"Balance between overhead and parallelism"
```

### Optimization Hierarchy

**Level 1: Algorithm** (10-100x gains)
- Choose right algorithm (O(n²) → O(n log n))

**Level 2: Memory Access** (2-50x gains)
- Coalescing, shared memory, minimize global touches

**Level 3: Compute Utilization** (2-10x gains)
- Tensor cores, occupancy, minimize divergence

**Level 4: System Integration** (1.5-5x gains)
- Minimize transfers, NVLink, InfiniBand, overlap

**Level 5: Micro-optimizations** (1.1-2x gains)
- Register allocation, instruction scheduling

**Rule**: "Optimize top-down. Don't micro-optimize a bad algorithm."

---

## 9. Interview Preparation Guide

### The Three-Question Framework

**1. What is slow?**
```bash
nvidia-smi              # Is GPU busy?
nsys profile ./app      # Where is time spent?
ncu ./app               # Which kernel?
```

**2. Why is it slow?**
```bash
ncu --metrics dram__throughput       # Memory-bound?
ncu --metrics sm__throughput         # Compute-bound?
ncu --metrics branch_efficiency      # Divergence?
ncu --metrics pipe_tensor_active     # Tensor cores?
```

**3. How do I fix it?**
```
Memory-bound → Tiling, caching, coalescing
Compute-bound → More parallelism, better algorithm
Divergence → Branchless code
Tensor cores → Mixed precision, align dimensions
```

### Answer Structure Template

**When asked about GPU optimization**:

1. **Problem**: What was slow?
2. **Diagnosis**: What was the bottleneck?
3. **Root Cause**: Why was it happening?
4. **Solution**: What did you do?
5. **Results**: Quantify the impact
6. **Learning**: Key takeaway

**Example**:
```
"In optimizing BERT inference:

Problem: 50ms latency per request (too slow for production)

Diagnosis: Profiling showed 80% DRAM bandwidth, 35% SM utilization
→ Memory-bound

Root Cause: Attention mechanism with poor memory access patterns

Solution:
- Fused attention kernel (reduced memory round-trips)
- Shared memory for QKV caching
- Enabled FP16 Tensor cores

Results:
- Latency: 50ms → 12ms (4.2x faster)
- Throughput: 20 → 80 QPS
- Memory BW: 80% → 65% (more efficient)
- SM util: 35% → 75% (balanced)

Learning: Memory optimization had bigger impact than compute optimization.
For memory-bound workloads, focus on access patterns before trying to
increase compute throughput."
```

### Mental Checklist

**Before Writing Code**:
- [ ] Can this run in parallel?
- [ ] What's the memory access pattern?
- [ ] How much data reuse?
- [ ] What's the arithmetic intensity?
- [ ] Do I need Tensor cores?

**After Writing Code**:
- [ ] Profile with nvidia-smi (GPU busy?)
- [ ] Profile with Nsight Systems (gaps in timeline?)
- [ ] Profile with Nsight Compute (bottleneck?)
- [ ] Check coalescing (>80%?)
- [ ] Check occupancy (60-80%?)
- [ ] Check Tensor cores (>70% for ML?)

**When Performance Poor**:
- [ ] Memory-bound? → Tile, cache, coalesce
- [ ] Compute-bound? → More parallelism
- [ ] Divergent? → Branchless code
- [ ] I/O bound? → Async, pinned memory
- [ ] Communication-bound? → NVLink, fusion

### Quick Reference Tables

**GPU Specs (A100 vs H100)**:
| | A100 | H100 |
|---|---|---|
| SMs | 108 | 132 |
| Tensor Cores | 432 | 528 |
| Memory | 40/80GB | 80GB |
| Memory BW | 2.0 TB/s | 3.35 TB/s |
| FP16 TFLOPS | 312 | 989 |

**Memory Hierarchy**:
| Level | Latency | Size (A100) |
|---|---|---|
| Registers | 1 cycle | 256KB/SM |
| Shared Mem | ~20 cycles | 164KB/SM |
| L2 Cache | ~200 cycles | 40MB |
| Global | ~400 cycles | 40-80GB |

**Performance Targets**:
| Metric | Target | Warning |
|---|---|---|
| GPU Util | >80% | <50% |
| Memory BW | >70% | <40% |
| Tensor Core | >70% | <30% |
| Occupancy | 60-80% | <40% |
| Coalescing | >80% | <50% |

---

## Conclusion

This documentation covers the essential GPU architecture concepts for ML infrastructure engineering. The key takeaways:

1. **GPUs achieve performance through massive parallelism and latency hiding**, not by making individual operations faster

2. **Memory is almost always the bottleneck** - optimize access patterns before compute

3. **Hierarchy matters** - understand the relationships between GPU → SM → Warp → Thread and Registers → Shared → L2 → Global

4. **Profile systematically** - use nvidia-smi, Nsight Systems, Nsight Compute in that order to narrow down bottlenecks

5. **Think in terms of trade-offs** - occupancy vs resources, latency vs throughput, granularity vs overhead

For interviews, focus on:
- Deep understanding of concepts (not memorization)
- Connecting to real-world scenarios
- Quantifying impact with numbers
- Demonstrating systematic problem-solving

Remember: Your job is to **keep all execution units busy** and **minimize time waiting for memory**.

Good luck!

---

**Version**: 1.0  
**Last Updated**: 2026-01-31  
**Word Count**: ~15,000 words  
**Reading Time**: ~60 minutes
