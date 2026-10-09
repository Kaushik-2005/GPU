#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>

namespace {

bool checkCuda(cudaError_t status, const char* operation) {
    if (status == cudaSuccess) {
        return true;
    }

    std::fprintf(stderr, "%s failed: %s\n", operation, cudaGetErrorString(status));
    return false;
}

}

__global__ void matrixAdd(const float* B, const float* C, float* A, int rows, int cols) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < rows && col < cols) {
        const int i = row * cols + col;
        A[i] = B[i] + C[i];
    }
}

int main() {
    constexpr int rows = 3;
    constexpr int cols = 5;
    constexpr int elementCount = rows * cols;

    const std::size_t bytes = static_cast<std::size_t>(elementCount) * sizeof(float);

    float B[elementCount];
    float C[elementCount];
    float A[elementCount];
    float A_ref[elementCount];

    // Initialize inputs and calculate the CPU reference.
    for (int i = 0; i < elementCount; ++i) {
        B[i] = static_cast<float>(i);
        C[i] = static_cast<float>(2 * i);
    }

    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < cols; ++col) {
            const int i = row * cols + col;
            A_ref[i] = B[i] + C[i];
        }
    }

    float* d_B = nullptr;
    float* d_C = nullptr;
    float* d_A = nullptr;

    constexpr int blockWidth = 2;
    constexpr int blockHeight = 2;
    const dim3 blockSize(blockWidth, blockHeight);
    const dim3 gridSize(
        (cols + blockWidth - 1) / blockWidth,
        (rows + blockHeight - 1) / blockHeight
    );
    int exitCode = EXIT_FAILURE;
    bool resultsMatch = true;

    if (!checkCuda(cudaMalloc(reinterpret_cast<void**>(&d_B), bytes),
                   "cudaMalloc d_B")) {
        goto cleanup;
    }
    if (!checkCuda(cudaMalloc(reinterpret_cast<void**>(&d_C), bytes),
                   "cudaMalloc d_C")) {
        goto cleanup;
    }
    if (!checkCuda(cudaMalloc(reinterpret_cast<void**>(&d_A), bytes),
                   "cudaMalloc d_A")) {
        goto cleanup;
    }

    if (!checkCuda(cudaMemcpy(d_B, B, bytes, cudaMemcpyHostToDevice),
                   "copy B to device")) {
        goto cleanup;
    }
    if (!checkCuda(cudaMemcpy(d_C, C, bytes, cudaMemcpyHostToDevice),
                   "copy C to device")) {
        goto cleanup;
    }

    matrixAdd<<<gridSize, blockSize>>>(d_B, d_C, d_A, rows, cols);

    if (!checkCuda(cudaGetLastError(), "matrixAdd launch")) {
        goto cleanup;
    }
    if (!checkCuda(cudaDeviceSynchronize(), "matrixAdd execution")) {
        goto cleanup;
    }

    if (!checkCuda(cudaMemcpy(A, d_A, bytes, cudaMemcpyDeviceToHost),
                   "copy A to host")) {
        goto cleanup;
    }

    for (int i = 0; i < elementCount; ++i) {
        if (std::fabs(A[i] - A_ref[i]) > 1e-5f) {
            std::fprintf(stderr,
                         "Mismatch at index %d: GPU=%f, CPU=%f\n",
                         i, A[i], A_ref[i]);
            resultsMatch = false;
            break;
        }
    }

    if (!resultsMatch) {
        goto cleanup;
    }

    std::printf("Matrix addition passed for %d x %d elements.\n",
                rows, cols);
    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < cols; ++col) {
            std::printf("%6.1f", A[row * cols + col]);
        }
        std::printf("\n");
    }

    exitCode = EXIT_SUCCESS;

cleanup:
    if (d_A != nullptr &&
        !checkCuda(cudaFree(d_A), "cudaFree d_A")) {
        exitCode = EXIT_FAILURE;
    }
    if (d_C != nullptr &&
        !checkCuda(cudaFree(d_C), "cudaFree d_C")) {
        exitCode = EXIT_FAILURE;
    }
    if (d_B != nullptr &&
        !checkCuda(cudaFree(d_B), "cudaFree d_B")) {
        exitCode = EXIT_FAILURE;
    }

    return exitCode;
}
