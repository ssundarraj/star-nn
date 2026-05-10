# star-nn C++ Scaffold

This directory is the C++ learning version of the algorithms in `../py`.

The current code is intentionally incomplete. Headers define the API, source
files contain default-return stubs, and tests describe the behavior to
implement.

The project targets C++23 so you can use standard range helpers such as
`std::views::zip`.

## Layout

- `include/star_nn/`: public headers
- `src/`: algorithm implementations
- `tests/`: assert-based learning tests
- `examples/`: small runnable entrypoints
- `Makefile`: explicit build commands

## Commands

```sh
make build
make test
make clean
```

`make test` is expected to fail until the TODO implementations are filled in.

## Suggested Order

1. Implement `knn.cpp` until `build/test_knn` passes.
2. Implement `annoy.cpp` until `build/test_annoy` passes.
3. Implement `hnsw.cpp` until `build/test_hnsw` passes.
4. Update `examples/main.cpp` as you want richer benchmarks.
