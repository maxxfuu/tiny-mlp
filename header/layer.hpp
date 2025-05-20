#pragma once
#include <vector>
#include "./neuron.hpp"

class Layer {
public:
  // Layer Constructor
  Layer(size_t num_neurons, size_t nin, bool nonlin = true) 
    : nin_(nin), nonlin_(nonlin) {
    neurons_.reserve(num_neurons);
    for (size_t i = 0; i < num_neurons; i++) {
      neurons_.push_back(Neuron(nin, nonlin));
    }
  }

  // Forward pass
  std::vector<Value> forward_pass(const std::vector<Value>& inputs) {
    std::vector<Value> outputs;
    outputs.reserve(neurons_.size());
    for (auto& neuron : neurons_) {
      outputs.push_back(neuron.forward_pass(inputs));
    }
    return outputs;
  }

private:
  std::vector<Neuron> neurons_;
  size_t nin_;
  bool nonlin_;
};