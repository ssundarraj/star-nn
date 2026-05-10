#pragma once

#include <cstddef>
#include <vector>

#include "star_nn/types.h"

namespace star_nn {

double euclidean_distance(const Vector &a, const Vector &b);
double manhattan_distance(const Vector &a, const Vector &b);

std::vector<std::size_t> find_k_nearest(const Vector &query,
                                        const Dataset &training_data,
                                        std::size_t k);

Label classify(const std::vector<std::size_t> &neighbor_indices,
               const Dataset &training_data);

Label predict(const Vector &query, const Dataset &training_data, std::size_t k);

double evaluate(const Dataset &test_data, const Dataset &training_data,
                std::size_t k);

} // namespace star_nn
