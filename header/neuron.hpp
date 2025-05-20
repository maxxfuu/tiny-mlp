#pragma once
#include <vector>
#include "./value.hpp"
#include <random>
#include <cmath>
class Neuron { 
public:
  // Neuron constructor
  Neuron(std::size_t nin, bool nonlin = true, Value bias = Value(0)) : weights_(nin), bias_(bias), nonlin_(nonlin) {
    // Generate random weights between -1 and 1
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-1.0, 1.0);
    for (std::size_t i = 0; i < nin; i++) {
      weights_[i] = Value(dis(gen));
    }
  }

  // Forward pass
  Value forward_pass(const std::vector<Value>& inputs) {
    inputs_ = inputs;
    Value output = Value(0);
    for (std::size_t i = 0; i < weights_.size(); i++) {
      output = output + weights_[i] * inputs_[i];
    }
    output = output + bias_;
    if (nonlin_) {
      return ReLU(output);
    }
    return output;
  }

private:
  std::vector<Value> weights_;
  std::vector<Value> inputs_;
  Value bias_;
  bool nonlin_; 
};