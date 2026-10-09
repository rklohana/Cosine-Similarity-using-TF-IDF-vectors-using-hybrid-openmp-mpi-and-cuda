#include "CudaCosineSimilarity.hpp"

#include <cuda_runtime.h>
#include <algorithm>
#include <climits>
#include <stdexcept>
#include <string>

namespace {
void check(cudaError_t result, const char* operation) {
    if (result != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(result));
    }
}

__global__ void cosineTile(const double* vectors, double* tile,
                           int rows, int columns, int firstRow, int tileRows) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int count = tileRows * rows;
    if (index >= count) return;
    const int a = firstRow + index / rows;
    const int b = index % rows;
    double dot = 0.0;
    for (int column = 0; column < columns; ++column) {
        dot += vectors[static_cast<std::size_t>(a) * columns + column] *
               vectors[static_cast<std::size_t>(b) * columns + column];
    }
    tile[index] = dot;
}
} // namespace

std::vector<double> cudaCosine(const std::vector<double>& vectors,
                               std::size_t rows, std::size_t columns) {
    if (!rows || !columns || vectors.size() != rows * columns) {
        throw std::invalid_argument("invalid TF-IDF matrix dimensions");
    }
    if (rows > static_cast<std::size_t>(INT_MAX) ||
        columns > static_cast<std::size_t>(INT_MAX) ||
        rows * std::min<std::size_t>(rows, 64) > static_cast<std::size_t>(INT_MAX)) {
        throw std::invalid_argument("matrix exceeds CUDA kernel index range");
    }

    constexpr std::size_t tileRows = 64;
    double* deviceVectors = nullptr;
    double* deviceTile = nullptr;
    std::vector<double> results(rows * rows);
    try {
        check(cudaMalloc(reinterpret_cast<void**>(&deviceVectors), vectors.size() * sizeof(double)), "cudaMalloc vectors");
        check(cudaMalloc(reinterpret_cast<void**>(&deviceTile),
                         std::min(rows, tileRows) * rows * sizeof(double)), "cudaMalloc tile");
        check(cudaMemcpy(deviceVectors, vectors.data(), vectors.size() * sizeof(double),
                         cudaMemcpyHostToDevice), "cudaMemcpy vectors");
        for (std::size_t first = 0; first < rows; first += tileRows) {
            const std::size_t current = std::min(tileRows, rows - first);
            const int count = static_cast<int>(current * rows);
            cosineTile<<<(count + 255) / 256, 256>>>(
                deviceVectors, deviceTile, static_cast<int>(rows),
                static_cast<int>(columns), static_cast<int>(first), static_cast<int>(current));
            check(cudaGetLastError(), "cosineTile launch");
            check(cudaMemcpy(results.data() + first * rows, deviceTile,
                             current * rows * sizeof(double), cudaMemcpyDeviceToHost),
                  "cudaMemcpy similarity tile");
        }
    } catch (...) {
        cudaFree(deviceTile);
        cudaFree(deviceVectors);
        throw;
    }
    check(cudaFree(deviceTile), "cudaFree tile");
    check(cudaFree(deviceVectors), "cudaFree vectors");
    return results;
}
