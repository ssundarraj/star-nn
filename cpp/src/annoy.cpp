#include "star_nn/annoy.h"
#include "star_nn/knn.h"

#include <algorithm>
#include <cstddef>
#include <iostream>
#include <numeric>
#include <random>
#include <ranges>
#include <unordered_set>
#include <vector>

namespace star_nn {

using std::size_t;

double dot(const Vector &a, const Vector &b) {
  // a · b = Σ(a_i * b_i)
  double d = 0.0;
  for (const auto [ai, bi] : std::views::zip(a, b)) {
    d += ai * bi;
  }
  return d;
}

std::pair<Vector, double> make_split_hyperplane(const Vector &a,
                                                const Vector &b) {
  Vector normal = std::vector<double>();
  Vector midpoint = std::vector<double>();
  normal.reserve(a.size());
  midpoint.reserve(a.size());

  for (const auto [ai, bi] : std::views::zip(a, b)) {
    normal.push_back(bi - ai);
    midpoint.push_back((ai + bi) / 2.0);
  }
  const auto offset = dot(normal, midpoint);

  return {normal, offset};
}

AnnoyIndex::AnnoyIndex(std::size_t n_trees, std::size_t max_leaf_size)
    : n_trees_(n_trees), max_leaf_size_(max_leaf_size) {}

std::pair<std::vector<size_t>, std::vector<size_t>>
AnnoyIndex::split_over_hyperplane_(const std::vector<size_t> &indices,
                                   const Vector &normal, double offset) {
  auto left = std::vector<size_t>();
  auto right = std::vector<size_t>();
  left.reserve(indices.size());
  right.reserve(indices.size());

  for (const auto idx : indices) {
    const auto &point = this->training_data_[idx].features;
    const auto is_left = (dot(normal, point) - offset) <= 0;
    if (is_left)
      left.push_back(idx);
    else
      right.push_back(idx);
  }
  return {left, right};
}

std::unique_ptr<Node>
AnnoyIndex::build_tree_(const std::vector<size_t> &indices) {
  if (indices.size() <= this->max_leaf_size_) {
    auto node = std::make_unique<Node>();
    node->is_leaf = true;
    node->indices = indices;
    return node;
  }

  static std::mt19937 rng(std::random_device{}());
  std::uniform_int_distribution<std::size_t> dist(0, indices.size() - 1);
  const std::size_t pos_a = dist(rng);
  std::size_t pos_b = dist(rng);
  while (pos_b == pos_a) {
    pos_b = dist(rng);
  }
  const auto idx_a = indices[pos_a];
  const auto idx_b = indices[pos_b];

  const auto &point_a = training_data_[idx_a].features;
  const auto &point_b = training_data_[idx_b].features;

  const auto [normal, offset] = make_split_hyperplane(point_a, point_b);
  const auto [left_indices, right_indices] =
      split_over_hyperplane_(indices, normal, offset);

  if (left_indices.empty() || right_indices.empty()) {
    auto node = std::make_unique<Node>();
    node->is_leaf = true;
    node->indices = indices;
    return node;
  }

  auto left = build_tree_(left_indices);
  auto right = build_tree_(right_indices);

  auto node = std::make_unique<Node>();
  node->is_leaf = false;
  node->left = std::move(left);
  node->right = std::move(right);
  node->normal = normal;
  node->offset = offset;
  return node;
}

void AnnoyIndex::build(const Dataset &training_data) {
  this->training_data_ = training_data;

  std::vector<std::size_t> indices(training_data_.size());
  std::iota(indices.begin(), indices.end(), 0);

  for (size_t i = 0; i < n_trees_; ++i) {
    std::cout << "  building tree " << (i + 1) << "/" << n_trees_ << "...\n";
    auto root = build_tree_(indices);
    forest_.push_back(std::move(root));
  }
}

std::vector<size_t> AnnoyIndex::query_tree_(const Node *tree,
                                            const Vector &query) const {
  if (tree->is_leaf)
    return tree->indices;

  const auto is_left = (dot(tree->normal, query) - tree->offset) <= 0;
  if (is_left)
    return query_tree_(tree->left.get(), query);
  else
    return query_tree_(tree->right.get(), query);
}

std::vector<std::size_t> AnnoyIndex::query(const Vector &query,
                                           std::size_t k) const {
  (void)query;
  (void)k;
  std::unordered_set<std::size_t> candidate_set;

  for (const auto &tree : forest_) {
    auto indices = query_tree_(tree.get(), query);
    candidate_set.insert(indices.begin(), indices.end());
  }

  std::vector<std::size_t> candidates(candidate_set.begin(),
                                      candidate_set.end());
  std::vector<std::pair<double, std::size_t>> scored_candidates;

  for (const auto c : candidates) {
    scored_candidates.push_back(
        {squared_euclidean_distance(training_data_[c].features, query), c});
  }

  std::partial_sort(scored_candidates.begin(),
                    scored_candidates.begin() +
                        std::min(k, scored_candidates.size()),
                    scored_candidates.end());

  candidates.clear();
  for (const auto &[dist, idx] : scored_candidates) {
    candidates.push_back(idx);
  }

  if (candidates.size() > k) {
    candidates.resize(k);
  }

  return candidates;
}

double AnnoyIndex::evaluate(const Dataset &test_data, std::size_t k) const {
  std::size_t correct = 0;
  const std::size_t progress_every =
      std::max<std::size_t>(1, test_data.size() / 20);

  for (std::size_t i = 0; i < test_data.size(); ++i) {
    const auto &dp = test_data[i];
    const auto neighbors = query(dp.features, k);
    if (classify(neighbors, training_data_) == dp.label) {
      ++correct;
    }

    if ((i + 1) % progress_every == 0 || i + 1 == test_data.size()) {
      std::cout << "  evaluated " << (i + 1) << "/" << test_data.size()
                << " queries...\n";
    }
  }

  return static_cast<double>(correct) / test_data.size();
}

std::size_t AnnoyIndex::n_trees() const { return n_trees_; }

std::size_t AnnoyIndex::max_leaf_size() const { return max_leaf_size_; }

const std::vector<std::unique_ptr<Node>> &AnnoyIndex::forest() const {
  return forest_;
}

} // namespace star_nn
