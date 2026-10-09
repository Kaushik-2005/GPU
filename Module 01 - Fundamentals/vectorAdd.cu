#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

bool checkCuda(cudaError_t status, const char* operation) {
    if (status == cudaSuccess) {
        return true;
    }

    std::fprintf(stderr, "%s failed: %s\n", operation,
                 cudaGetErrorString(status));
    return false;
}

}

__global__ void vectorAdd(const float* A, const float* B, float* C, int N) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        C[i] = A[i] + B[i];
    }
}

int main() {
    constexpr int N = 10;
    constexpr int blockSize = 256;
    const int gridSize = (N + blockSize - 1) / blockSize;
    const std::size_t bytes = static_cast<std::size_t>(N) * sizeof(float);

    std::vector<float> A(N);
    std::vector<float> B(N);
    std::vector<float> C(N);
    std::vector<float> C_ref(N);
    for (int i = 0; i < N; ++i) {
        A[i] = static_cast<float>(i);
        B[i] = static_cast<float>(2 * i);
    }

    for (int i = 0; i < N; ++i) {
        C_ref[i] = A[i] + B[i];
    }

    float* d_A = nullptr;
    float* d_B = nullptr;
    float* d_C = nullptr;
    int exitCode = EXIT_FAILURE;
    bool resultsMatch = true;

    if (!checkCuda(cudaMalloc(reinterpret_cast<void**>(&d_A), bytes),
                   "cudaMalloc d_A")) {
        goto cleanup;
    }
    if (!checkCuda(cudaMalloc(reinterpret_cast<void**>(&d_B), bytes),
                   "cudaMalloc d_B")) {
        goto cleanup;
    }
    if (!checkCuda(cudaMalloc(reinterpret_cast<void**>(&d_C), bytes),
                   "cudaMalloc d_C")) {
        goto cleanup;
    }

    if (!checkCuda(cudaMemcpy(d_A, A.data(), bytes, cudaMemcpyHostToDevice),
                   "copy A to device")) {
        goto cleanup;
    }
    if (!checkCuda(cudaMemcpy(d_B, B.data(), bytes, cudaMemcpyHostToDevice),
                   "copy B to device")) {
        goto cleanup;
    }

    vectorAdd<<<gridSize, blockSize>>>(d_A, d_B, d_C, N);
    if (!checkCuda(cudaGetLastError(), "vectorAdd launch")) {
        goto cleanup;
    }
    if (!checkCuda(cudaDeviceSynchronize(), "vectorAdd execution")) {
        goto cleanup;
    }

    if (!checkCuda(cudaMemcpy(C.data(), d_C, bytes, cudaMemcpyDeviceToHost),
                   "copy C to host")) {
        goto cleanup;
    }

    for (int i = 0; i < N; ++i) {
        if (std::fabs(C[i] - C_ref[i]) > 1e-5f) {
            std::fprintf(stderr,
                         "Mismatch at index %d: GPU=%f, CPU=%f\n",
                         i, C[i], C_ref[i]);
            resultsMatch = false;
            break;
        }
    }

    if (!resultsMatch) {
        goto cleanup;
    }

    std::printf("Vector addition passed for %d elements.\n", N);
    std::printf("Per-thread GPU results:\n");
    for (int i = 0; i < N; ++i) {
        const int blockIndex = i / blockSize;
        const int threadIndex = i % blockSize;
        std::printf(
            "blockIdx.x=%d, threadIdx.x=%d -> i=%d: %.1f + %.1f = %.1f\n",
            blockIndex, threadIndex, i, A[i], B[i], C[i]);
    }

    exitCode = EXIT_SUCCESS;

cleanup:
    if (d_C != nullptr &&
        !checkCuda(cudaFree(d_C), "cudaFree d_C")) {
        exitCode = EXIT_FAILURE;
    }
    if (d_B != nullptr &&
        !checkCuda(cudaFree(d_B), "cudaFree d_B")) {
        exitCode = EXIT_FAILURE;
    }
    if (d_A != nullptr &&
        !checkCuda(cudaFree(d_A), "cudaFree d_A")) {
        exitCode = EXIT_FAILURE;
    }

    return exitCode;
}
