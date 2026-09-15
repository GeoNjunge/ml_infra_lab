import time
import random

def make_zero_matrix(r, c): return [[0.0 for _ in range(c)] for _ in range(r)]
def make_random_matrix(r, c): return [[random.random() for _ in range(c)] for _ in range(r)]
def transpose(M): return [[M[r][c] for r in range(len(M))] for c in range(len(M[0]))]


def matmul(M1, M2):
    zero_mat = make_zero_matrix(len(M1), len(M2[0]))
    for i in range(len(M1)):
        for j in range(len(M2[0])):
            for k in range(len(M1[0])):
                zero_mat[i][j] += M1[i][k] * M2[k][j]

    return zero_mat


def matadd(A, B):
    return [[A[i][j] + B[i][j] for j in range(len(A[0]))] for i in range(len(A))]


class MatrixValue:
    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self.rows = len(data)
        self.cols = len(data[0]) if self.rows > 0 else 0
        self.grad = make_zero_matrix(self.rows, self.cols)
        self._prev = set(_children)
        self._backward = lambda: None

    def __matmul__(self, other):
        out = MatrixValue(matmul(self.data, other.data), (self, other), '@')
        def _backward():
            grad_A = matmul(out.grad, transpose(other.data))
            grad_B = matmul(transpose(self.data), out.grad)
            for i in range(self.rows):
                for j in range(self.cols): self.grad[i][j] += grad_A[i][j]
            for i in range(other.rows):
                for j in range(other.cols): other.grad[i][j] += grad_B[i][j]
        out._backward = _backward
        return out
        
    def __add__(self, other):
        out = MatrixValue(matadd(self.data, other.data), (self, other), '+')
        def _backward():
            for i in range(self.rows):
                for j in range(self.cols):
                    self.grad[i][j] += out.grad[i][j]
                    other.grad[i][j] += out.grad[i][j]
        out._backward = _backward
        return out
    
    def backward(self):
        visited, topo = set(), []
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                    topo.append(v)
        build_topo(self)
        self.grad = [[1.0 for _ in range(self.cols)] for _ in range(self.rows)]
        for node in reversed(topo):
            node._backward()

                

