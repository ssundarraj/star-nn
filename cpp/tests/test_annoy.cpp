#include <cassert>
#include <cmath>
#include <filesystem>
#include <iostream>

#include "star_nn/annoy.h"

using star_nn::Dataset;

int main() {
  std::cout << "Testing dot product...\n";
  assert(star_nn::dot({1.0, 0.0}, {1.0, 0.0}) == 1.0);
  assert(star_nn::dot({1.0, 2.0}, {3.0, 4.0}) == 11.0);
  assert(star_nn::dot({0.0, 0.0}, {5.0, 5.0}) == 0.0);

  std::cout << "Testing split hyperplane math...\n";
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

  std::cout << "Testing Annoy constructor...\n";
  star_nn::AnnoyIndex index(5, 3);
  assert(index.n_trees() == 5);
  assert(index.max_leaf_size() == 3);

  std::cout << "Testing Annoy build...\n";
  index.build(data);
  assert(index.forest().size() == 5);

  std::cout << "Testing Annoy query near A cluster...\n";
  const auto neighbors_a = index.query({0.05, 0.05}, 3);
  assert(neighbors_a.size() == 3);
  for (const auto neighbor : neighbors_a) {
    assert(neighbor < 5);
  }

  std::cout << "Testing Annoy query near B cluster...\n";
  const auto neighbors_b = index.query({10.05, 10.05}, 3);
  for (const auto neighbor : neighbors_b) {
    assert(neighbor >= 5);
  }

  std::cout << "Testing Annoy save/load...\n";
  const std::filesystem::path index_path = "/tmp/star_nn_annoy_test.bin";
  index.save(index_path.string());
  assert(std::filesystem::exists(index_path));
  assert(std::filesystem::file_size(index_path) > 0);

  star_nn::AnnoyIndex loaded;
  loaded.load(index_path.string(), data);
  assert(loaded.n_trees() == index.n_trees());
  assert(loaded.max_leaf_size() == index.max_leaf_size());
  assert(loaded.query({0.05, 0.05}, 3) == neighbors_a);
  assert(loaded.query({10.05, 10.05}, 3) == neighbors_b);

  std::cout << "Testing Annoy mmap load...\n";
  star_nn::AnnoyIndex mapped;
  mapped.load_mmap(index_path.string(), data);
  assert(mapped.n_trees() == index.n_trees());
  assert(mapped.max_leaf_size() == index.max_leaf_size());
  assert(mapped.query({0.05, 0.05}, 3) == neighbors_a);
  assert(mapped.query({10.05, 10.05}, 3) == neighbors_b);

  std::cout << "Annoy tests passed.\n";
}
