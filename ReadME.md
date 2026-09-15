# Scaled Neural Network Layer Benchmark
We will be profiling a single forward and backward pass of a deep neural network layer.
**Forward Propagation**:
    $${Output} = (\mathbf{X} \mathbin{@} \mathbf{W}) + \mathbf{B}$$

**BackPropagation**:
If `C = A × B`, then:
    $$\frac{dl}{dA} = \frac{dl}{dC} * \mathbf{B^T}$$ $$\frac{dl}{dB} = \mathbf{A^T} * \frac{dl}{dC}$$

Where 
`X` is a **1 × 1000** vector, and 
`W` is a **1000 × 1000** matrix. This forces the autograd graph to process exactly 1 million matrix parameters inside the graph structure.

## Step 1: Pure python baseline [autograd.py](./python/autograd.py)

Ensures scaling up safely without Numpy by initializing a MatrixValue engine that initalizes parameters cleanly as a list of lists.
Uses the `i-j-k` loop for matrix multiplication. No optimizations

Run command
```bash
python3 profiler.py
```

## Step 2: Unoptimized C++ porting [naive_autograd.cpp](./cpp/naive_autograd.cpp)
Mirrors the unoptimized python program using structs. Space is allocated using 1D vectors mapped as 2D grids $${index} = {i} * {cols} + {j}$$
This keeps layouts smooth. Also compiles with *0* optimizations *-O0*

Run command
```bash
g++ -O0 autograd_naive.cpp -o naive_autograd
./naive_autograd
```

## Step 3: Cache-Friendly C++ Optimization [cached_autograd.cpp](./cpp/cached_autograd.cpp)
Uses the `i-j-k` loop permutation for matrix multiplication. Backpropagation has a hidden transposition, so changing traversal order reduces latency on large operations

### Why is this better than the naive approach
Accessing `B[k][j]` in the naive approach forces processor to skip memory rows leading to frequent CPU cache misses.
Reordering to `i-k-j` aligns memory accesses with the sequential RAM structure. 

Run Command
```bash
g++ -O0 autograd_cache.cpp -o cache_autograd
./cache_autograd
```

## Step 4: Multi-Threaded SIMD Engine [autograd_max.cpp](./cpp/autograd_max.cpp)
Uses multithreading and vector instructions via OpenMP. OpenMP splits the rows pf the matrix multiplication across every active thread core of the processor

Run Command
```bash
g++ -O3 -march=native -fopenmp autograd_max.cpp -o max_autograd
./max_autograd
```