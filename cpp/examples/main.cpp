#include <iostream>

#include "star_nn/annoy.h"
#include "star_nn/hnsw.h"
#include "star_nn/knn.h"

int main() {
  const star_nn::Dataset data = {
      {{0.0, 0.0}, "A"},
      {{1.0, 1.0}, "A"},
      {{10.0, 10.0}, "B"},
      {{11.0, 11.0}, "B"},
  };

  std::cout << "KNN prediction: " << star_nn::predict({0.5, 0.5}, data, 3)
            << '\n';

  star_nn::AnnoyIndex annoy;
  annoy.build(data);
  std::cout << "Annoy neighbors: " << annoy.query({0.5, 0.5}, 3).size() << '\n';

  star_nn::HNSWIndex hnsw;
  hnsw.build(data);
  std::cout << "HNSW neighbors: " << hnsw.query({0.5, 0.5}, 3).size() << '\n';
}
