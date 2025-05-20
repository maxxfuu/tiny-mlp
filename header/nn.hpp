#pragma once
#include <vector>
#include "./engine.hpp"
#include <random>
#include <cmath>

class Module {
public:
  virtual std::vector<Value*> parameters();
  virtual void zero_grad();
  virtual ~Module() = default;
};

class Neuron { 
public:
  // Neuron constructor
  Neuron(std::size_t nin, bool nonlin = true, Value bias = Value(0, [](){}, "")) : weights_(nin), bias_(bias), nonlin_(nonlin) {
    // Generate random weights between -1 and 1
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-1.0, 1.0);
    for (std::size_t i = 0; i < nin; i++) {
      weights_[i] = Value(dis(gen), [](){}, "weight");
    }
  }

  // Forward pass
  Value forward_pass(const std::vector<Value>& inputs) {
    inputs_ = inputs;
    Value output = Value(0, [](){}, "output");
    for (std::size_t i = 0; i < weights_.size(); i++) {
      output.data += weights_[i].data * inputs_[i].data;
    }
    output.data += bias_.data;
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