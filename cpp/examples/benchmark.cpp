#include "star_nn/annoy.h"
#include "star_nn/datasets.h"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

struct BenchmarkParams {
  std::size_t n_trees;
  std::size_t max_leaf_size;
};

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
  const std::vector<BenchmarkParams> params = {
      {5, 5},  {5, 10},  {5, 20},  {10, 5}, {10, 10},
      {10, 20}, {20, 5}, {20, 10}, {20, 20},
  };
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

  const std::filesystem::path results_path = "benchmarks/results.csv";
  std::filesystem::create_directories(results_path.parent_path());
  std::ofstream results(results_path);
  results << "dataset,train_rows,test_rows,dims,n_trees,max_leaf_size,k,"
             "build_ms,build_ms_per_tree,save_ms,load_index_ms,index_bytes,"
             "eval_ms,eval_ms_per_sample,mmap_load_ms,mmap_eval_ms,"
             "mmap_eval_ms_per_sample,mmap_accuracy,"
             "accuracy\n";

  for (std::size_t i = 0; i < params.size(); ++i) {
    const auto [n_trees, max_leaf_size] = params[i];
    std::cout << "\nBenchmark " << (i + 1) << "/" << params.size()
              << ": n_trees=" << n_trees
              << " max_leaf_size=" << max_leaf_size << " k=" << k << '\n';

    std::cout << "Building Annoy index...\n";
    start = Clock::now();
    star_nn::AnnoyIndex annoy(n_trees, max_leaf_size);
    annoy.build(train);
    const auto annoy_build_ms = millis_since(start);

    const std::filesystem::path index_path =
        "benchmarks/annoy_index_" + std::to_string(n_trees) + "_" +
        std::to_string(max_leaf_size) + ".bin";

    std::cout << "Saving Annoy index...\n";
    start = Clock::now();
    annoy.save(index_path.string());
    const auto annoy_save_ms = millis_since(start);
    const auto index_bytes = std::filesystem::file_size(index_path);

    std::cout << "Loading Annoy index...\n";
    start = Clock::now();
    star_nn::AnnoyIndex loaded_annoy;
    loaded_annoy.load(index_path.string(), train);
    const auto annoy_load_index_ms = millis_since(start);

    std::cout << "Memory-mapping Annoy index...\n";
    start = Clock::now();
    star_nn::AnnoyIndex mmap_annoy;
    mmap_annoy.load_mmap(index_path.string(), train);
    const auto annoy_mmap_load_ms = millis_since(start);

    std::cout << "Running Annoy evaluation from loaded index...\n";
    start = Clock::now();
    const auto annoy_accuracy = loaded_annoy.evaluate(test, k);
    const auto annoy_eval_ms = millis_since(start);

    std::cout << "Running Annoy evaluation from mmap index...\n";
    start = Clock::now();
    const auto annoy_mmap_accuracy = mmap_annoy.evaluate(test, k);
    const auto annoy_mmap_eval_ms = millis_since(start);

    const auto build_ms_per_tree = annoy_build_ms / n_trees;
    const auto eval_ms_per_sample = annoy_eval_ms / test.size();
    const auto mmap_eval_ms_per_sample = annoy_mmap_eval_ms / test.size();

    std::cout << "Annoy: accuracy=" << annoy_accuracy * 100.0
              << "% build=" << annoy_build_ms
              << "ms build/tree=" << build_ms_per_tree
              << "ms save=" << annoy_save_ms
              << "ms load/index=" << annoy_load_index_ms
              << "ms index_bytes=" << index_bytes
              << " query/eval=" << annoy_eval_ms
              << "ms eval/sample=" << eval_ms_per_sample
              << "ms mmap_load=" << annoy_mmap_load_ms
              << "ms mmap_eval=" << annoy_mmap_eval_ms
              << "ms mmap_eval/sample=" << mmap_eval_ms_per_sample
              << "ms mmap_accuracy=" << annoy_mmap_accuracy * 100.0
              << "%\n";

    results << dataset_name(train_path, test_path) << ',' << train.size() << ','
            << test.size() << ',' << train[0].features.size() << ','
            << n_trees << ',' << max_leaf_size << ',' << k << ','
            << annoy_build_ms << ',' << build_ms_per_tree << ','
            << annoy_save_ms << ',' << annoy_load_index_ms << ','
            << index_bytes << ','
            << annoy_eval_ms << ',' << eval_ms_per_sample << ','
            << annoy_mmap_load_ms << ',' << annoy_mmap_eval_ms << ','
            << mmap_eval_ms_per_sample << ',' << annoy_mmap_accuracy << ','
            << annoy_accuracy << '\n';
  }

  std::cout << "Wrote latest benchmark results to " << results_path << '\n';
}
