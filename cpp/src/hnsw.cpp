#include "star_nn/hnsw.h"

namespace star_nn {

HNSWIndex::HNSWIndex(std::size_t m, std::size_t ef_construction,
                     std::size_t ef_search)
    : m_(m), ef_construction_(ef_construction), ef_search_(ef_search) {}

std::size_t HNSWIndex::m() const { return m_; }

std::size_t HNSWIndex::ef_construction() const { return ef_construction_; }

std::size_t HNSWIndex::ef_search() const { return ef_search_; }

std::optional<std::size_t> HNSWIndex::entry_point() const {
  return entry_point_;
}

std::size_t HNSWIndex::max_level() const { return max_level_; }

const Dataset &HNSWIndex::data() const { return data_; }

int HNSWIndex::random_level() const { return 0; }

std::vector<std::pair<double, std::size_t>>
HNSWIndex::search_layer(const Vector &query,
                        const std::vector<std::size_t> &entry_points,
                        std::size_t ef, std::size_t layer) const {
  (void)query;
  (void)entry_points;
  (void)ef;
  (void)layer;
  return {};
}

std::vector<std::size_t> HNSWIndex::select_neighbors(
    const std::vector<std::pair<double, std::size_t>> &candidates,
    std::size_t m) const {
  (void)candidates;
  (void)m;
  return {};
}

void HNSWIndex::insert(const Vector &features, const Label &label) {
  (void)features;
  (void)label;
}

void HNSWIndex::build(const Dataset &training_data) { (void)training_data; }

std::vector<std::size_t> HNSWIndex::query(const Vector &query,
                                          std::size_t k) const {
  (void)query;
  (void)k;
  return {};
}

double HNSWIndex::evaluate(const Dataset &test_data, std::size_t k) const {
  (void)test_data;
  (void)k;
  return 0.0;
}

} // namespace star_nn
