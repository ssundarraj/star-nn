#include "star_nn/annoy.h"

namespace star_nn {

double dot(const Vector &a, const Vector &b) {
  (void)a;
  (void)b;
  return 0.0;
}

std::pair<Vector, double> make_split_hyperplane(const Vector &point_a,
                                                const Vector &point_b) {
  (void)point_a;
  (void)point_b;
  return {{}, 0.0};
}

AnnoyIndex::AnnoyIndex(std::size_t n_trees, std::size_t max_leaf_size)
    : n_trees_(n_trees), max_leaf_size_(max_leaf_size) {}

void AnnoyIndex::build(const Dataset &training_data) { (void)training_data; }

std::vector<std::size_t> AnnoyIndex::query(const Vector &query,
                                           std::size_t k) const {
  (void)query;
  (void)k;
  return {};
}

double AnnoyIndex::evaluate(const Dataset &test_data, std::size_t k) const {
  (void)test_data;
  (void)k;
  return 0.0;
}

std::size_t AnnoyIndex::n_trees() const { return n_trees_; }

std::size_t AnnoyIndex::max_leaf_size() const { return max_leaf_size_; }

const std::vector<std::unique_ptr<Node>> &AnnoyIndex::forest() const {
  return forest_;
}

} // namespace star_nn
