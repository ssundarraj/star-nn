#pragma once

#include <string>
#include <vector>

namespace star_nn {

using Vector = std::vector<double>;
using Label = std::string;

struct DataPoint {
  Vector features;
  Label label;
};

using Dataset = std::vector<DataPoint>;

} // namespace star_nn
