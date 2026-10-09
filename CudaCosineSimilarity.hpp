#pragma once

#include <cstddef>
#include <vector>

// Input rows are unit-normalized, row-major TF-IDF vectors.
// Empty documents have all-zero rows and therefore zero similarity.
std::vector<double> cudaCosine(const std::vector<double>& vectors,
                               std::size_t rows, std::size_t columns);
