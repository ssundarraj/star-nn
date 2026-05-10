#pragma once

#include <cstddef>
#include <memory>
#include <utility>
#include <vector>

#include "star_nn/types.h"

namespace star_nn {

struct Node {
  bool is_leaf = true;
  std::vector<std::size_t> indices;
  std::unique_ptr<Node> left;
  std::unique_ptr<Node> right;
  Vector normal;
  double offset = 0.0;
};

double dot(const Vector &a, const Vector &b);
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
  const std::vector<std::unique_ptr<Node>> &forest() const;

private:
  std::size_t n_trees_;
  std::size_t max_leaf_size_;
  std::vector<std::unique_ptr<Node>> forest_;
  Dataset training_data_;
};

} // namespace star_nn
