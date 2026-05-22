#pragma once

#include <string>

#include "star_nn/types.h"

namespace star_nn {

Dataset load_csv(const std::string &path);

} // namespace star_nn
