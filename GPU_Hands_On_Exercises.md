# GPU Architecture - Hands-On Exercises
## Practical Labs to Master GPU Programming

**Time Required**: 10-15 hours total  
**Prerequisites**: NVIDIA GPU access, CUDA toolkit, PyTorch

---

## Quick Start Setup

```bash
# Verify GPU
nvidia-smi

# Install tools
pip install torch torchvision matplotlib numpy

# Create workspace
mkdir gpu-labs && cd gpu-labs
```

---

## Lab 1: GPU Architecture Discovery (1 hour)

**Goal**: Understand your GPU's capabilities

### Exercise 1.1: Query GPU Properties

```python
import torch

print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Compute Capability: {torch.cuda.get_device_capability(0)}")
print(f"Total Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
print(f"SMs: {torch.cuda.get_device_properties(0).multi_processor_count}")
print(f"Max Threads/SM: {torch.cuda.get_device_properties(0).max_threads_per_multi_processor}")
```

**Questions**:
1. How many SMs does your GPU have?
2. What's the max theoretical occupancy?
3. How much global memory?

---

## Lab 2: Memory Hierarchy Benchmarking (2 hours)

**Goal**: Measure memory latency and bandwidth

### Exercise 2.1: Memory Bandwidth Test

```python
import torch
import time

def benchmark_memory(size_mb, iterations=100):
    size = size_mb * 1024 * 1024 // 4  # floats
    src = torch.randn(size, device='cuda')
    dst = torch.empty_like(src)
    
    torch.cuda.synchronize()
    start = time.time()
    
    for _ in range(iterations):
        dst.copy_(src)
    
    torch.cuda.synchronize()
    elapsed = time.time() - start
    
    bandwidth = (size * 4 * 2 * iterations) / elapsed / 1e9  # GB/s
    return bandwidth

# Test different sizes
for size_mb in [1, 10, 100, 1000]:
    bw = benchmark_memory(size_mb)
    print(f"{size_mb:4d} MB: {bw:6.2f} GB/s")
```

---

## Lab 3: Tensor Core Activation (2 hours)

**Goal**: Enable and verify Tensor Core usage

### Exercise 3.1: FP32 vs FP16 Performance

```python
import torch
import torch.nn as nn
from torch.cuda.amp import autocast
import time

class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1024, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 1000)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

model = SimpleNet().cuda()
x = torch.randn(256, 1024, device='cuda')

# FP32 baseline
torch.cuda.synchronize()
start = time.time()
for _ in range(100):
    y = model(x)
torch.cuda.synchronize()
fp32_time = time.time() - start

# FP16 with Tensor Cores
torch.cuda.synchronize()
start = time.time()
for _ in range(100):
    with autocast():
        y = model(x)
torch.cuda.synchronize()
fp16_time = time.time() - start

print(f"FP32: {fp32_time*10:.2f} ms")
print(f"FP16: {fp16_time*10:.2f} ms")
print(f"Speedup: {fp32_time/fp16_time:.2f}x")
```

---

## Lab 4: Memory Coalescing (1.5 hours)

**Goal**: Understand access pattern impact

### Exercise 4.1: SoA vs AoS

```python
import torch
import time

# Array of Structs (bad for GPU)
class ParticlesAoS:
    def __init__(self, n):
        self.data = torch.randn(n, 6, device='cuda')
    
    def update(self, dt):
        self.data[:, :3] += self.data[:, 3:] * dt

# Struct of Arrays (good for GPU)  
class ParticlesSoA:
    def __init__(self, n):
        self.pos = torch.randn(n, 3, device='cuda')
        self.vel = torch.randn(n, 3, device='cuda')
    
    def update(self, dt):
        self.pos += self.vel * dt

n = 1_000_000
dt = 0.01

aos = ParticlesAoS(n)
soa = ParticlesSoA(n)

# Benchmark AoS
torch.cuda.synchronize()
start = time.time()
for _ in range(1000):
    aos.update(dt)
torch.cuda.synchronize()
aos_time = time.time() - start

# Benchmark SoA
torch.cuda.synchronize()
start = time.time()
for _ in range(1000):
    soa.update(dt)
torch.cuda.synchronize()
soa_time = time.time() - start

print(f"AoS: {aos_time*1000:.2f} ms")
print(f"SoA: {soa_time*1000:.2f} ms")
print(f"Speedup: {aos_time/soa_time:.2f}x")
```

---

## Lab 5: Profiling Workflow (2 hours)

**Goal**: Master profiling tools

### Exercise 5.1: Profile Pipeline

```python
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.fc = nn.Linear(128 * 32 * 32, 1000)
    
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        return self.fc(x)

model = Model().cuda()
data = torch.randn(32, 3, 32, 32, device='cuda')

output = model(data)
loss = output.sum()
loss.backward()
```

**Profile**:
```bash
nsys profile -o timeline python script.py
ncu --set full -o analysis python script.py
```

---

## Lab 6: Production Optimization (3 hours)

**Goal**: End-to-end optimization

### Exercise 6.1: Optimized Training

```python
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader, TensorDataset

# Efficient DataLoader
dataset = TensorDataset(
    torch.randn(1000, 3, 224, 224),
    torch.randint(0, 1000, (1000,))
)

dataloader = DataLoader(
    dataset,
    batch_size=128,
    num_workers=4,
    pin_memory=True,
    prefetch_factor=2
)

# Model and training
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1)
        )
        self.fc = nn.Linear(128 * 224 * 224, 1000)
    
    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

model = Model().cuda()
optimizer = torch.optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()
scaler = GradScaler()

import time
start = time.time()

for data, target in dataloader:
    data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
    
    optimizer.zero_grad()
    
    with autocast():
        output = model(data)
        loss = criterion(output, target)
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

torch.cuda.synchronize()
print(f"Time: {time.time() - start:.2f}s")
```

---

## Quick Reference

**Profiling Commands**:
```bash
nvidia-smi -l 1
nsys profile -o timeline python script.py
ncu --metrics sm__throughput.avg.pct_of_peak_sustained_active python script.py
```

**Performance Targets**:
- GPU Util: >80%
- Tensor Core: >70%
- FP16 Speedup: >3x

---

**Total Time**: 10-15 hours  
**Difficulty**: Intermediate  

Good luck! 🚀
