#include <cassert>
#include <cmath>
#include <iostream>
#include <ranges>
#include <set>
#include <vector>

#include "star_nn/knn.h"

using star_nn::DataPoint;
using star_nn::Dataset;
using star_nn::Vector;

int main() {
  std::cout << "Testing Euclidean distance...\n";
  assert(star_nn::euclidean_distance({0.0, 0.0}, {0.0, 0.0}) == 0.0);
  assert(star_nn::euclidean_distance({0.0, 0.0}, {3.0, 4.0}) == 5.0);
  assert(star_nn::euclidean_distance({1.0}, {4.0}) == 3.0);

  const double d1 =
      star_nn::euclidean_distance({1.0, 2.0, 3.0}, {4.0, 5.0, 6.0});
  const double d2 =
      star_nn::euclidean_distance({4.0, 5.0, 6.0}, {1.0, 2.0, 3.0});
  assert(std::abs(d1 - d2) < 1e-9);

  std::cout << "Testing Manhattan distance...\n";
  assert(star_nn::manhattan_distance({0.0, 0.0}, {0.0, 0.0}) == 0.0);
  assert(star_nn::manhattan_distance({0.0, 0.0}, {3.0, 4.0}) == 7.0);
  assert(star_nn::manhattan_distance({1.0, 2.0}, {4.0, 6.0}) == 7.0);

  std::cout << "Testing find_k_nearest...\n";
  const Dataset data = {
      {{0.0, 0.0}, "A"}, {{1.0, 0.0}, "A"}, {{10.0, 10.0}, "B"},
      {{0.5, 0.5}, "A"}, {{9.0, 9.0}, "B"},
  };

  const auto neighbors = star_nn::find_k_nearest({0.0, 0.0}, data, 3);
  assert(neighbors.size() == 3);
  assert(std::set<std::size_t>(neighbors.begin(), neighbors.end()) ==
         std::set<std::size_t>({0, 1, 3}));

  std::cout << "Testing classify...\n";
  const Dataset data2 = {
      {{0.0, 0.0}, "cat"},
      {{1.0, 0.0}, "cat"},
      {{2.0, 0.0}, "dog"},
      {{3.0, 0.0}, "cat"},
  };

  assert(star_nn::classify({0, 1, 2}, data2) == "cat");
  assert(star_nn::classify({2}, data2) == "dog");

  std::cout << "Testing predict...\n";
  const Dataset data3 = {
      {{0.0, 0.0}, "A"},   {{1.0, 1.0}, "A"},   {{2.0, 2.0}, "A"},
      {{10.0, 10.0}, "B"}, {{11.0, 11.0}, "B"}, {{12.0, 12.0}, "B"},
  };

  assert(star_nn::predict({0.5, 0.5}, data3, 3) == "A");
  assert(star_nn::predict({10.5, 10.5}, data3, 3) == "B");

  std::cout << "KNN tests passed.\n";
}
