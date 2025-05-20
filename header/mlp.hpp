#pragma once
#include <vector>
#include "./layer.hpp"

class MLP {
public:
  // MLP Constructor
  MLP(const std::vector<size_t>& sizes) {
    for (size_t i = 0; i < sizes.size() - 1; i++) {
      // All layers except the last one use nonlinearity
      bool is_last_layer = (i == sizes.size() - 2);
      layers_.push_back(Layer(sizes[i + 1], sizes[i], !is_last_layer));
    }
  }

  // Forward pass
  std::vector<Value> forward_pass(const std::vector<Value>& inputs) {
    std::vector<Value> outputs = inputs;
    for (auto& layer : layers_) {
      outputs = layer.forward_pass(outputs);
    }
    return outputs;
  }

private:
  std::vector<Layer> layers_;
};