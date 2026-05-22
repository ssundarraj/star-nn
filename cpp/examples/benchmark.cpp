#include "star_nn/annoy.h"
#include "star_nn/datasets.h"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>

namespace {

using Clock = std::chrono::steady_clock;

double millis_since(Clock::time_point start) {
  const auto elapsed = Clock::now() - start;
  return std::chrono::duration<double, std::milli>(elapsed).count();
}

std::string dataset_name(const std::string &train_path,
                         const std::string &test_path) {
  if (train_path.contains("mnist") || test_path.contains("mnist")) {
    return "mnist";
  }
  if (train_path.contains("iris")) {
    return "iris";
  }
  return "custom";
}

} // namespace

int main(int argc, char **argv) {
  const std::size_t k = 5;
  const std::size_t n_trees = 10;
  const std::size_t max_leaf_size = 10;
  const std::string train_path = argc > 1 ? argv[1] : "../data/mnist_train.csv";
  const std::string test_path = argc > 2 ? argv[2] : "../data/mnist_test.csv";

  std::cout << std::fixed << std::setprecision(2);

  std::cout << "Loading CSV data...\n";
  auto start = Clock::now();
  const auto train_source = star_nn::load_csv(train_path);
  const auto test_source =
      test_path.empty() ? star_nn::Dataset{} : star_nn::load_csv(test_path);
  const auto load_ms = millis_since(start);

  star_nn::Dataset train;
  star_nn::Dataset test;
  if (test_path.empty()) {
    const auto split = train_source.size() * 80 / 100;
    train =
        star_nn::Dataset(train_source.begin(), train_source.begin() + split);
    test = star_nn::Dataset(train_source.begin() + split, train_source.end());
  } else {
    train = train_source;
    test = test_source;
  }

  std::cout << "Train CSV: " << train_path << '\n';
  if (!test_path.empty()) {
    std::cout << "Test CSV: " << test_path << '\n';
  }
  std::cout << "Rows: train=" << train.size() << " test=" << test.size()
            << " dims=" << train[0].features.size() << '\n';
  std::cout << "Load: " << load_ms << "ms\n";

  std::cout << "Building Annoy index...\n";
  start = Clock::now();
  star_nn::AnnoyIndex annoy(n_trees, max_leaf_size);
  annoy.build(train);
  const auto annoy_build_ms = millis_since(start);

  std::cout << "Running Annoy evaluation...\n";
  start = Clock::now();
  const auto annoy_accuracy = annoy.evaluate(test, k);
  const auto annoy_eval_ms = millis_since(start);

  std::cout << "Annoy: accuracy=" << annoy_accuracy * 100.0
            << "% build=" << annoy_build_ms << "ms query/eval=" << annoy_eval_ms
            << "ms\n";

  const std::filesystem::path results_path = "benchmarks/results.csv";
  std::filesystem::create_directories(results_path.parent_path());
  std::ofstream results(results_path);
  results << "dataset,train_rows,test_rows,dims,n_trees,max_leaf_size,k,"
             "build_ms,eval_ms,accuracy\n";
  results << dataset_name(train_path, test_path) << ',' << train.size() << ','
          << test.size() << ',' << train[0].features.size() << ',' << n_trees
          << ',' << max_leaf_size << ',' << k << ',' << annoy_build_ms << ','
          << annoy_eval_ms << ',' << annoy_accuracy << '\n';
  std::cout << "Wrote latest benchmark results to " << results_path << '\n';
}
