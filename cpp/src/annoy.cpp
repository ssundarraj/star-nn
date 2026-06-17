#include "star_nn/annoy.h"
#include "star_nn/knn.h"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <ranges>
#include <span>
#include <stdexcept>
#include <unordered_set>
#include <vector>

namespace star_nn {

using std::size_t;

double dot(const Vector &a, const Vector &b) {
  return dot(std::span<const double>{a}, std::span<const double>{b});
}

double dot(std::span<const double> a, std::span<const double> b) {
  assert(a.size() == b.size());

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

std::size_t AnnoyIndex::build_tree_(const std::vector<size_t> &indices) {
  if (indices.size() <= this->max_leaf_size_) {
    std::size_t leaf_item_offset = leaf_items_.size();
    leaf_items_.insert(leaf_items_.end(), indices.begin(), indices.end());

    auto node = DiskNode{
        .left_index = -1,  // Leaf
        .right_index = -1, // Leaf
        .leaf_item_offset = leaf_item_offset,
        .leaf_item_count = indices.size(),
        .normal_offset = 0,
        .hyperplane_offset = 0.0,
    };

    nodes_.push_back(node);
    return nodes_.size() - 1;
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
    std::size_t leaf_item_offset = leaf_items_.size();
    leaf_items_.insert(leaf_items_.end(), indices.begin(), indices.end());

    auto node = DiskNode{
        .left_index = -1,  // Leaf
        .right_index = -1, // Leaf
        .leaf_item_offset = leaf_item_offset,
        .leaf_item_count = indices.size(),
        .normal_offset = 0,
        .hyperplane_offset = 0.0,
    };

    nodes_.push_back(node);
    return nodes_.size() - 1;
  }

  const auto left = build_tree_(left_indices);
  const auto right = build_tree_(right_indices);

  const auto normal_offset = normals_.size();
  normals_.insert(normals_.end(), normal.begin(), normal.end());
  auto node = DiskNode{
      .left_index = static_cast<std::int64_t>(left),
      .right_index = static_cast<std::int64_t>(right),

      .leaf_item_offset = 0,
      .leaf_item_count = 0,
      .normal_offset = normal_offset,
      .hyperplane_offset = offset,
  };

  nodes_.push_back(node);
  return nodes_.size() - 1;
}

void AnnoyIndex::build(const Dataset &training_data) {
  this->training_data_ = training_data;
  this->dims_ = training_data_[0].features.size();

  std::vector<std::size_t> indices(training_data_.size());
  std::iota(indices.begin(), indices.end(), 0);

  for (size_t i = 0; i < n_trees_; ++i) {
    std::cout << "  building tree " << (i + 1) << "/" << n_trees_ << "...\n";
    auto root = build_tree_(indices);
    forest_.push_back(root);
  }
}

template <typename T>
void write_vector(std::ofstream &out, const std::vector<T> &v) {
  out.write(reinterpret_cast<const char *>(v.data()),
            static_cast<std::streamsize>(v.size() * sizeof(T)));
}

template <typename T> void read_vector(std::ifstream &in, std::vector<T> &v) {
  in.read(reinterpret_cast<char *>(v.data()),
          static_cast<std::streamsize>(v.size() * sizeof(T)));
  if (!in) {
    throw std::runtime_error("failed to read Annoy index section");
  }
}

void AnnoyIndex::save(const std::string &path) const {
  std::ofstream out(path, std::ios::binary);

  AnnoyFileHeader header = {
      .version = 1,
      .dims = dims_,
      .n_trees = n_trees_,
      .max_leaf_size = max_leaf_size_,
      .forest_count = forest_.size(),
      .node_count = nodes_.size(),
      .leaf_item_count = leaf_items_.size(),
      .normal_count = normals_.size(),
  };

  out.write(reinterpret_cast<const char *>(&header), sizeof(header));

  write_vector(out, forest_);
  write_vector(out, nodes_);
  write_vector(out, leaf_items_);
  write_vector(out, normals_);
}
void AnnoyIndex::load(const std::string &path, const Dataset &training_data) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    throw std::runtime_error("failed to open Annoy index file: " + path);
  }

  AnnoyFileHeader header;
  in.read(reinterpret_cast<char *>(&header), sizeof(header));
  if (!in) {
    throw std::runtime_error("failed to read Annoy index header");
  }

  if (header.version != 1) {
    throw std::runtime_error("unsupported Annoy index version");
  }

  if (training_data.empty()) {
    throw std::runtime_error("cannot load Annoy index with empty training data");
  }

  if (header.dims != training_data[0].features.size()) {
    throw std::runtime_error("Annoy index dimensions do not match training data");
  }

  dims_ = header.dims;
  n_trees_ = header.n_trees;
  max_leaf_size_ = header.max_leaf_size;
  training_data_ = training_data;

  forest_.resize(header.forest_count);
  nodes_.resize(header.node_count);
  leaf_items_.resize(header.leaf_item_count);
  normals_.resize(header.normal_count);

  read_vector(in, forest_);
  read_vector(in, nodes_);
  read_vector(in, leaf_items_);
  read_vector(in, normals_);
}

std::span<const size_t> AnnoyIndex::query_tree_(const size_t node_idx,
                                                const Vector &query) const {
  DiskNode tree = nodes_[node_idx];
  if (tree.left_index == -1 && tree.right_index == -1) {
    return std::span{
        leaf_items_.data() + tree.leaf_item_offset,
        tree.leaf_item_count,
    };
  }

  const auto normal = std::span{
      normals_.data() + tree.normal_offset,
      dims_,
  };
  const auto is_left = (dot(normal, query) - tree.hyperplane_offset) <= 0;
  if (is_left)
    return query_tree_(tree.left_index, query);
  else
    return query_tree_(tree.right_index, query);
}

std::vector<std::size_t> AnnoyIndex::query(const Vector &query,
                                           std::size_t k) const {
  std::unordered_set<std::size_t> candidate_set;

  for (const auto root_idx : forest_) {
    auto indices = query_tree_(root_idx, query);
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

const std::vector<std::size_t> &AnnoyIndex::forest() const { return forest_; }

} // namespace star_nn
