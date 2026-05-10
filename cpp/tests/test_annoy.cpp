#include <cassert>
#include <cmath>

#include "star_nn/annoy.h"

using star_nn::Dataset;

int main() {
  assert(star_nn::dot({1.0, 0.0}, {1.0, 0.0}) == 1.0);
  assert(star_nn::dot({1.0, 2.0}, {3.0, 4.0}) == 11.0);
  assert(star_nn::dot({0.0, 0.0}, {5.0, 5.0}) == 0.0);

  const auto [normal, offset] =
      star_nn::make_split_hyperplane({0.0, 0.0}, {2.0, 0.0});
  assert(std::abs(star_nn::dot(normal, {1.0, 0.0}) - offset) < 1e-9);
  assert(star_nn::dot(normal, {0.0, 0.0}) - offset <= 0.0);
  assert(star_nn::dot(normal, {2.0, 0.0}) - offset >= 0.0);

  const Dataset data = {
      {{0.0, 0.0}, "A"},   {{0.1, 0.0}, "A"},   {{0.0, 0.1}, "A"},
      {{0.2, 0.1}, "A"},   {{0.1, 0.2}, "A"},   {{10.0, 10.0}, "B"},
      {{10.1, 10.0}, "B"}, {{10.0, 10.1}, "B"}, {{10.2, 10.1}, "B"},
      {{10.1, 10.2}, "B"},
  };

  star_nn::AnnoyIndex index(5, 3);
  assert(index.n_trees() == 5);
  assert(index.max_leaf_size() == 3);

  index.build(data);
  assert(index.forest().size() == 5);

  const auto neighbors_a = index.query({0.05, 0.05}, 3);
  assert(neighbors_a.size() == 3);
  for (const auto neighbor : neighbors_a) {
    assert(neighbor < 5);
  }

  const auto neighbors_b = index.query({10.05, 10.05}, 3);
  for (const auto neighbor : neighbors_b) {
    assert(neighbor >= 5);
  }
}
