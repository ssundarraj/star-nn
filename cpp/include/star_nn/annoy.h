#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <utility>
#include <vector>

#include "star_nn/types.h"

namespace star_nn {

struct DiskNode {
  int64_t left_index;        // -1 if leaf
  int64_t right_index;       // -1 if leaf
  uint64_t leaf_item_offset; // offset into leaf_items
  uint64_t leaf_item_count;  // >0 for leaf
  uint64_t normal_offset;    // offset into float/double array
  double hyperplane_offset;  // hyperplane offset
};

double dot(const Vector &a, const Vector &b);
double dot(std::span<const double> a, std::span<const double> b);
std::pair<Vector, double> make_split_hyperplane(const Vector &point_a,
                                                const Vector &point_b);

class AnnoyIndex {
public:
  AnnoyIndex(std::size_t n_trees = 10, std::size_t max_leaf_size = 10);

  void build(const Dataset &training_data);
  std::vector<std::size_t> query(const Vector &query, std::size_t k) const;
  double evaluate(const Dataset &test_data, std::size_t k) const;

  std::size_t n_trees() const;
  std::size_t max_leaf_size() const;

  const std::vector<size_t> &forest() const;

private:
  std::vector<DiskNode> nodes_;
  std::vector<std::size_t> forest_;     // index of root nodes
  std::vector<std::size_t> leaf_items_; // index of items in leaf nodes
  std::vector<double> normals_;

  std::size_t dims_ = 0;

  std::size_t build_tree_(
      const std::vector<size_t> &indices); // returns index of node in nodes

  std::pair<std::vector<size_t>, std::vector<size_t>>
  split_over_hyperplane_(const std::vector<size_t> &indices,
                         const Vector &normal, double offset);

  std::span<const size_t> query_tree_(const size_t node_idx,
                                      const Vector &query) const;

  std::size_t n_trees_;
  std::size_t max_leaf_size_;
  Dataset training_data_;
};

} // namespace star_nn
