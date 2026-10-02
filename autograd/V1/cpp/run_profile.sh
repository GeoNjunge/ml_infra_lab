#!/bin/bash

# Clear any old data rows before starting run
echo "Implementation, Time_nanoseconds" > native_benchmarks.csv

echo "========  STARTING NAIVE SYSTEM PROFILING ======="

# 1. Profile the Naive Engine
echo "Compiling and running Naive baseline..."
g++ -o0 naive_autograd.cpp -o naive_bin
./naive_bin

# 2. Profile the Cache Friendly Layout Engine
echo "Compiling and running Cache-Friendly engine..."
g++ -O0 cached_autograd.cpp -o cache_bin
./cache_bin

# 3. Profile the Max Optimized Parallel Engine
echo "Compiling and running Aggressive Multi-Threaded engine..."
g++ -O3 -march=native -fopenmp autograd_max.cpp -o max_bin
./max_bin