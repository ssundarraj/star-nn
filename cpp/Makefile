CXX ?= c++
CXXFLAGS := -std=c++23 -Wall -Wextra -pedantic -Iinclude
BUILD_DIR := build
BUILD_STAMP := $(BUILD_DIR)/.dir

LIB_SRCS := src/knn.cpp src/annoy.cpp src/hnsw.cpp
LIB_OBJS := $(LIB_SRCS:src/%.cpp=$(BUILD_DIR)/%.o)

TEST_SRCS := tests/test_knn.cpp tests/test_annoy.cpp tests/test_hnsw.cpp
TEST_BINS := $(TEST_SRCS:tests/%.cpp=$(BUILD_DIR)/%)

EXAMPLE_SRCS := examples/main.cpp
EXAMPLE_BINS := $(EXAMPLE_SRCS:examples/%.cpp=$(BUILD_DIR)/%)

.PHONY: all build test clean

all: build

build: $(TEST_BINS) $(EXAMPLE_BINS)

test: $(TEST_BINS)
	$(BUILD_DIR)/test_knn
	$(BUILD_DIR)/test_annoy
	$(BUILD_DIR)/test_hnsw

$(BUILD_STAMP):
	mkdir -p $(BUILD_DIR)
	touch $(BUILD_STAMP)

$(BUILD_DIR)/%.o: src/%.cpp | $(BUILD_STAMP)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/test_%: tests/test_%.cpp $(LIB_OBJS) | $(BUILD_STAMP)
	$(CXX) $(CXXFLAGS) $< $(LIB_OBJS) -o $@

$(BUILD_DIR)/main: examples/main.cpp $(LIB_OBJS) | $(BUILD_STAMP)
	$(CXX) $(CXXFLAGS) $< $(LIB_OBJS) -o $@

clean:
	rm -rf $(BUILD_DIR)
