#include <cassert>

#include "star_nn/hnsw.h"

using star_nn::Dataset;

int main() {
  star_nn::HNSWIndex empty_index(4, 20, 10);
  assert(empty_index.m() == 4);
  assert(empty_index.ef_construction() == 20);
  assert(empty_index.ef_search() == 10);
  assert(!empty_index.entry_point().has_value());

  star_nn::HNSWIndex level_index(4);
  for (int i = 0; i < 1000; ++i) {
    assert(level_index.random_level() >= 0);
  }

  const Dataset data = {
      {{0.0, 0.0}, "A"},   {{0.1, 0.0}, "A"},   {{0.0, 0.1}, "A"},
      {{0.2, 0.1}, "A"},   {{0.1, 0.2}, "A"},   {{10.0, 10.0}, "B"},
      {{10.1, 10.0}, "B"}, {{10.0, 10.1}, "B"}, {{10.2, 10.1}, "B"},
      {{10.1, 10.2}, "B"},
  };

  star_nn::HNSWIndex index(4, 20, 10);
  index.build(data);
  assert(index.data().size() == 10);
  assert(index.entry_point().has_value());

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
