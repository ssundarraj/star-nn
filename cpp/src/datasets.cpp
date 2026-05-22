#include "star_nn/datasets.h"

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace star_nn {

Dataset load_csv(const std::string &path) {
  std::ifstream file(path);
  if (!file) {
    throw std::runtime_error("failed to open CSV: " + path);
  }

  Dataset data;
  std::string line;
  while (std::getline(file, line)) {
    if (line.empty()) {
      continue;
    }

    std::stringstream ss(line);
    std::string cell;
    std::vector<std::string> cells;
    while (std::getline(ss, cell, ',')) {
      cells.push_back(cell);
    }

    if (cells.size() < 2) {
      continue;
    }

    Vector features;
    features.reserve(cells.size() - 1);
    for (std::size_t i = 0; i + 1 < cells.size(); ++i) {
      features.push_back(std::stod(cells[i]));
    }

    data.push_back({features, cells.back()});
  }

  return data;
}

} // namespace star_nn
