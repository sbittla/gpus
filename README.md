# GPU Architecture Documentation

This comprehensive documentation covers GPU architecture concepts for ML infrastructure engineers.

## Files Included

1. **gpu_architecture_part1_core.md** - Core compute architecture (GPU, SM, CUDA cores, Tensor cores, Warps, Thread blocks)
2. **gpu_architecture_part2_execution.md** - Execution model (Kernels, Occupancy)
3. **gpu_architecture_part3_memory.md** - Complete memory hierarchy (Registers, Shared memory, L2, Global memory, Coalescing)
4. **gpu_architecture_part4_distributed.md** - Multi-GPU systems (NVLink, InfiniBand)
5. **gpu_architecture_part5_profiling.md** - Profiling and observability tools
6. **gpu_architecture_part6_kubernetes.md** - Kubernetes integration and cloud deployment
7. **gpu_architecture_part7_optimization.md** - Performance optimization strategies
8. **gpu_architecture_part8_relationships.md** - Concept relationships and mental models

## How to Use

Each file builds on previous concepts. Read in order for best understanding.

For interviews: Focus on sections 1-3 (core architecture and memory) as these form the foundation.

## Quick Navigation

- **Basics**: Start with Part 1 (Core Architecture)
- **Performance**: Jump to Part 7 (Optimization) after understanding Parts 1-3
- **Production**: Parts 4, 5, 6 for infrastructure/production concerns
- **Interview Prep**: Part 8 for frameworks and mental models
