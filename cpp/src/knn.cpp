#include "star_nn/knn.h"

namespace star_nn {

double euclidean_distance(const Vector &a, const Vector &b) {
  (void)a;
  (void)b;
  return 0.0;
}

double manhattan_distance(const Vector &a, const Vector &b) {
  (void)a;
  (void)b;
  return 0.0;
}

std::vector<std::size_t> find_k_nearest(const Vector &query,
                                        const Dataset &training_data,
                                        std::size_t k) {
  (void)query;
  (void)training_data;
  (void)k;
  return {};
}

Label classify(const std::vector<std::size_t> &neighbor_indices,
               const Dataset &training_data) {
  (void)neighbor_indices;
  (void)training_data;
  return "";
}

Label predict(const Vector &query, const Dataset &training_data,
              std::size_t k) {
  (void)query;
  (void)training_data;
  (void)k;
  return "";
}

double evaluate(const Dataset &test_data, const Dataset &training_data,
                std::size_t k) {
  (void)test_data;
  (void)training_data;
  (void)k;
  return 0.0;
}

} // namespace star_nn
