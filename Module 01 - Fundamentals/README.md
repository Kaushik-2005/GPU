# Module 01 — GPU Fundamentals

## Concepts Learned

### Introduction

- **Heterogeneous computing** combines CPUs and GPUs. A CPU uses complex control logic and large caches to keep a few threads responsive; a GPU devotes more hardware to high throughput across many threads.
- A GPU fits problems with many independent operations over a large data set. Small workloads, sequential dependencies, and data-transfer costs can make CPU execution preferable.
- A GPU is organized into **streaming multiprocessors (SMs)** that schedule groups of threads. Many ready threads let the GPU hide long memory or arithmetic latency by doing other work while some threads wait.
- Application speedup is limited by its sequential portion. If fraction `p` can be parallelized and that part is sped up by `s`, overall speedup is
  `S = 1 / ((1 - p) + p / s)`,
  where `p` is the parallelizable fraction and `s` is the speedup of that portion.
- Memory bandwidth and data delivery can limit performance even when plenty of parallel work exists. CPUs and GPUs often work together, each handling the parts suited to its design.
- Parallel programming models differ in how they express work: **MPI** passes messages between distributed nodes; **OpenMP** uses directives for shared-memory CPU parallelism; **OpenACC** uses directives for accelerator offload; **OpenCL** provides a cross-vendor compute model; **CUDA** exposes NVIDIA GPU execution and memory controls.

### History of GPU Computing

- Early GPUs used fixed-function stages for graphics rendering. Programmable vertex shaders transform vertices, geometry shaders operate on primitives, and pixel shaders shade pixels.
- Vertex and pixel operations often use independent data, exposing parallel work. Unified processors let the GPU share execution resources across programmable graphics stages.
- **GPGPU** means general-purpose computing on GPUs. Early GPGPU programs had to encode computations as graphics work: inputs were textures, shaders did the calculations, and outputs were pixel values in a frame buffer. General memory writes and data types were restricted.
- CUDA introduced a direct compute model with general memory loads and stores, a hierarchy of threads, synchronization, and atomic operations, plus its compiler, libraries, and runtime.
- A CUDA program can expose more threads than the GPU can run at once. Blocks are scheduled onto available hardware, allowing the same program to run on GPUs with different amounts of execution capacity.

### Introduction to Data Parallelism and CUDA C

- **Data parallelism** applies the same operation to different data elements. **Task parallelism** runs independent kinds of work at the same time. Large data-parallel workloads are a common source of GPU scalability.
- CUDA uses **SPMD (single program, multiple data)**: threads execute the same kernel code but use different indices and data.
- A CUDA program has **host** code (CPU) and **device** code (GPU). A **kernel** is a device function launched from the host; its launch creates a grid of blocks, each containing threads.
- `__global__` declares a kernel, `__device__` a device function, and `__host__` a host function. Unqualified functions are host functions by default.
- A **warp** is a group of 32 threads scheduled together on NVIDIA GPUs. When lanes in a warp take different branch paths, the paths may execute separately with some lanes inactive.
- In a one-dimensional grid, a thread’s global data index is `blockIdx.x * blockDim.x + threadIdx.x`. The grid replaces the CPU loop: each thread handles its assigned iteration. A bounds check such as `i < N` protects data when the final block contains extra threads.
- One-dimensional indexing example with four threads per block:
  ```text
  blockIdx.x:   0                 1
  threadIdx.x:  0  1  2  3       0  1  2  3
  global i:     0  1  2  3       4  5  6  7
  ```
- For 2D work, `col = blockIdx.x * blockDim.x + threadIdx.x` and `row = blockIdx.y * blockDim.y + threadIdx.y`; flatten row-major coordinates with `i = row * cols + col`.
- A `3 x 5` row-major matrix maps to linear indices as follows:
  ```text
  row 0:  0   1   2   3   4
  row 1:  5   6   7   8   9
  row 2: 10  11  12  13  14
  ```
- For the matrix-add kernel's `2 x 2` blocks, grid x advances across columns and grid y advances across rows. Its bounds check skips threads whose row or column lies outside the `3 x 5` matrix.
- In the basic memory model, host and device memory are separate. `cudaMalloc` allocates device bytes, `cudaMemcpy` copies data in a specified direction, and `cudaFree` releases device allocations. Inputs move to the device before the kernel; results return to the host afterward. Free buffers once no queued GPU work needs them so device memory can be reused.
- CUDA API calls return error status. Check allocation and copy calls; check a kernel launch with `cudaGetLastError()`. Kernel launches are asynchronous with respect to the host, and `cudaDeviceSynchronize()` waits for device work and reports execution errors when an explicit wait is needed.

## CUDA Kernels Implemented

### vectorAdd.cu

- **Purpose:** Compute `C[i] = A[i] + B[i]` and compare the GPU output with a CPU reference.
- **Thread mapping:** One thread computes one element using `i = blockIdx.x * blockDim.x + threadIdx.x`.
- **Bounds:** The kernel writes only when `i < N`, protecting arrays if the launch includes extra threads.
- **Host and device work:** The host initializes `A[i] = i` and `B[i] = 2*i`, builds a CPU reference, allocates device arrays, copies inputs with `cudaMemcpyHostToDevice`, launches the kernel, checks launch and execution errors, copies `C` back with `cudaMemcpyDeviceToHost`, compares with absolute tolerance `1e-5`, then frees device buffers.
- **Current launch:** `N = 10`, `blockSize = 256`, and `gridSize = 1`. Threads 0–9 compute results; threads 10–255 return at the bounds check. The host prints each valid thread's block index, thread index, input values, and sum.
- The 10-element vector addition passed against the CPU reference before the million-element comparison exercise.

### matrixAdd.cu

- **Purpose:** Add two row-major `3 x 5` matrices and compare the GPU result with a CPU reference.
- **Thread mapping:** Each thread computes one `(row, col)` pair using the 2D block and thread coordinates. The flattened array index is `row * cols + col`.
- **Bounds:** A `2 x 2` block shape and a `3 x 2` grid launch 24 threads; the `row < rows && col < cols` check lets only the 15 in-range threads write results.
- **Host and device work:** The host initializes both inputs and the CPU reference, allocates and copies device buffers, launches the kernel, checks launch and execution errors, copies the output back, compares with tolerance `1e-5`, and frees device memory.
- The `3 x 5` run passed; the output was `0, 3, 6, ... 42` in row-major order.
