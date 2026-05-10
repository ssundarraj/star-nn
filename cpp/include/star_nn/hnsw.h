#pragma once

#include <cstddef>
#include <optional>
#include <unordered_map>
#include <utility>
#include <vector>

#include "star_nn/types.h"

namespace star_nn {

class HNSWIndex {
public:
  HNSWIndex(std::size_t m = 16, std::size_t ef_construction = 200,
            std::size_t ef_search = 50);

  std::size_t m() const;
  std::size_t ef_construction() const;
  std::size_t ef_search() const;
  std::optional<std::size_t> entry_point() const;
  std::size_t max_level() const;
  const Dataset &data() const;

  int random_level() const;
  std::vector<std::pair<double, std::size_t>>
  search_layer(const Vector &query,
               const std::vector<std::size_t> &entry_points, std::size_t ef,
               std::size_t layer) const;
  std::vector<std::size_t> select_neighbors(
      const std::vector<std::pair<double, std::size_t>> &candidates,
      std::size_t m) const;

  void insert(const Vector &features, const Label &label);
  void build(const Dataset &training_data);
  std::vector<std::size_t> query(const Vector &query, std::size_t k) const;
  double evaluate(const Dataset &test_data, std::size_t k) const;

private:
  std::size_t m_;
  std::size_t ef_construction_;
  std::size_t ef_search_;
  Dataset data_;
  std::vector<std::unordered_map<std::size_t, std::vector<std::size_t>>>
      layers_;
  std::optional<std::size_t> entry_point_;
  std::size_t max_level_ = 0;
};

} // namespace star_nn
