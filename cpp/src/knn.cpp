#include "star_nn/knn.h"

#include <cmath>
#include <cstddef>
#include <ranges>
#include <unordered_map>
#include <utility>

namespace star_nn {

double euclidean_distance(const Vector &a, const Vector &b) {
  double d = 0.0;
  for (auto [x, y] : std::views::zip(a, b)) {
    d += std::pow(x - y, 2);
  }
  return std::sqrt(d);
}

double manhattan_distance(const Vector &a, const Vector &b) {
  double d = 0.0;
  for (auto [x, y] : std::views::zip(a, b)) {
    d += std::abs(x - y);
  }
  return d;
}

std::vector<std::size_t> find_k_nearest(const Vector &query,
                                        const Dataset &training_data,
                                        std::size_t k) {

  std::vector<std::pair<double, size_t>> candidates;
  for (std::size_t i = 0; i < training_data.size(); ++i) {
    const auto &dp = training_data[i];

    auto dist = euclidean_distance(dp.features, query);

    if (candidates.size() < k || candidates.back().first > dist) {
      auto item = std::make_pair(dist, i);

      auto pos = std::lower_bound(
          candidates.begin(), candidates.end(), item,
          [](const auto &a, const auto &b) { return a.first < b.first; });
      candidates.insert(pos, item);

      if (candidates.size() > k) {
        candidates.pop_back(); // remove farthest
      }
      continue;
    }
  }

  // return candidates
  std::vector<size_t> output(k);
  for (std::size_t i = 0; i < candidates.size(); ++i) {
    output[i] = candidates[i].second;
  }
  return output;
}

Label classify(const std::vector<std::size_t> &neighbor_indices,
               const Dataset &training_data) {
  std::unordered_map<Label, int> counts;

  int max_count = -1;
  Label max_label = "";

  for (const auto i : neighbor_indices) {
    const auto label = training_data[i].label;

    counts[label] += 1;
    if (counts[label] > max_count) {
      max_count = counts[label];
      max_label = label;
    }
  }

  return max_label;
}

Label predict(const Vector &query, const Dataset &training_data,
              std::size_t k) {
  const auto k_nearest = find_k_nearest(query, training_data, k);
  const auto label = classify(k_nearest, training_data);

  return label;
}

double evaluate(const Dataset &test_data, const Dataset &training_data,
                std::size_t k) {
  std::size_t correct = 0;

  for (const auto &dp : test_data) {
    if (predict(dp.features, training_data, k) == dp.label) {
      ++correct;
    }
  }

  return static_cast<double>(correct) / test_data.size();
}

} // namespace star_nn
