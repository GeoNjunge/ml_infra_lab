from autograd.python.autograd import MatrixValue, make_random_matrix
import time

N = 1000
print(f"Creating {N}x{N} computation graph...")
X = MatrixValue(make_random_matrix(1, N))     # 1x1000 Input Vector
W = MatrixValue(make_random_matrix(N, N))     # 1000x1000 Weights Matrix
B = MatrixValue(make_random_matrix(1, N))     # 1x1000 Bias Vector

t0 = time.time()
# Forward Pass
Y = (X @ W) + B
t1 = time.time()

# Backward Pass
Y.backward()
t2 = time.time()

with open("native_benchmarks.csv", "a") as fs:
    fs.write(f"Naive_python_code {t2 - t0:.4f} seconds\n")
    
print(f"Python Forward Pass:  {t1 - t0:.4f} seconds")
print(f"Python Backward Pass: {t2 - t1:.4f} seconds")
print(f"Total Time:           {t2 - t0:.4f} seconds")
