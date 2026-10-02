#include <iostream>
#include <vector>
#include <chrono>
#include <memory>
#include <fstream>

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

// Naive i-j-k matmul
shared_ptr<MatrixValue> matmul_naive(shared_ptr<MatrixValue> A, shared_ptr<MatrixValue> B) {
    auto out = make_shared<MatrixValue>(A->rows, B->cols, 0.0);
    out->prev = {A, B};

    for(int i=0; i<A->rows; ++i){
        for(int j=0; j<B->cols; ++j){
            for(int k=0; k<A->cols; ++k) {
                out->data[i * out->cols + j] += A->data[i * A->cols + k] * B->data[k * B->cols + j];           
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
    int size = 1000;
    auto A = make_shared<MatrixValue>(1, size, 1.0);
    auto B = make_shared<MatrixValue>(size, size, 1.0);

    auto start = steady_clock::now();
    
    auto out = matmul_naive(A, B);
    out->grad = vector<double>(out->rows * out->cols, 1.0); // Seed output grad
    out->backward_fn(out.get()); // Trigger actual math execution
    
    auto end = steady_clock::now();
    auto time_taken = duration_cast<milliseconds>(end - start); // Using nanoseconds for precision

    // Open a local file stream in Append Mode
    ofstream report_file;
    report_file.open("native_benchmarks.csv", ios_base::app);
    
    // Write out the configuration and precise runtime metric
    // Format: Implementation_Name, Time_In_Nanoseconds
    report_file << "Naive_i_j_k_Engine," << time_taken.count() /1000.0 << "\n";
    
    report_file.close();
    
cout << "Log complete. Latency: " << time_taken.count() << " secs" << endl;
    return 0;
}