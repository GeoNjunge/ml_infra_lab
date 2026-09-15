#include <iostream>
#include <vector>
#include <chrono>
#include <memory>

using namespace std;
using namespace std::chrono;

struct MatrixValue : enable_shared_from_this<MatrixValue> {
    vector<double> data;
    vector<double> grad;
    int rows, cols;
    vector<shared_ptr<MatrixValue>> prev;
    void (*backward_fn)(MatrixValue*){};

    MatrixValue(int r, int c, double val) : rows(r), cols(c) {
        data.resize(r * c, val);
        grad.resize(r * c, 0.0);
        backward_fn = [](MatrixValue*){};
    }
};

// SIMD optimized loop(OpenMP and Multithreading)
shared_ptr<MatrixValue> matmul_naive(shared_ptr<MatrixValue> A, shared_ptr<MatrixValue> B) {
    auto out = make_shared<MatrixValue>(A->rows, B->cols, 0.0);
    out->prev = {A, B};

    #pragma omp parallel for
    for(int i=0; i<A->rows; ++i){
        for(int k=0; k<A->cols; ++k) {
            auto r = A->data[i * A->cols + k];
            for(int j=0; j<B->cols; ++j){
                out->data[i * out->cols + j] += r * B->data[k * B->cols + j];           
            }
        }
    }

    out->backward_fn = [](MatrixValue* self) {
        auto A = self->prev[0];
        auto B = self->prev[1];

        // Backprop loop
        for(int i=0; i<A->rows; ++i) {
            for(int j=0; j<A->cols; ++j) {
                for(int k=0; k<self->cols; ++k) {
                    A->grad[i*A->cols + j] += self->grad[i*self->cols + k] * B->data[j*B->cols + k];
                }
            }
        }
    };
    return out;
}

int main() {
    // Initialize matrices A and B
    int size = 1000;
    auto A = make_shared<MatrixValue>(1, size, 1.0);
    auto B = make_shared<MatrixValue>(size, size, 1.0);

        // Profile the time taken
    auto start = steady_clock::now();
    // Compute the product
    auto out = matmul_naive(A, B);

    // Backpropagation loop
    auto back_prop = out->backward_fn;
    auto end = steady_clock::now();
    auto time_taken = duration_cast<nanoseconds>(end - start);

    cout << "Time taken: " << time_taken.count() << "nanoseconds" << endl;
    return 0;
}