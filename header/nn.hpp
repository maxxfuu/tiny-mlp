#pragma once
#include <vector>
#include "./engine.hpp"
#include <random>
#include <cmath>

class Module {
public:
  virtual std::vector<Value*> parameters() = 0;
  virtual void zero_grad() = 0;
  virtual ~Module() = default;
};

class Neuron : public Module {
public:
  // Neuron constructor
  Neuron(std::size_t nin, bool nonlin = true, Value bias = Value(0, {}, [](){}, "")) : weights_(nin), bias_(bias), nonlin_(nonlin) {
    // Xavier/Glorot initialization
    std::random_device rd;
    std::mt19937 gen(rd());
    double limit = std::sqrt(6.0 / (nin + 1.0)); // +1 for the output
    std::uniform_real_distribution<> dis(-limit, limit);
    for (std::size_t i = 0; i < nin; i++) {
      weights_[i] = Value(dis(gen), {}, [](){}, "weight");
    }
  }

  // Forward pass
  Value forward_pass(const std::vector<Value>& inputs) {
    inputs_ = inputs;
    Value act = Value(0, {}, [](){}, "act");
    for (std::size_t i = 0; i < weights_.size(); i++) {
      Value temp = (weights_[i] * inputs_[i]);
      act = act + temp;
    }
    act = act + bias_;
    if (nonlin_) {
      return ReLU(act);
    }
    return act;
  }

  std::vector<Value*> parameters() override {
    std::vector<Value*> params;
    for (auto& w : weights_) {
      params.push_back(&w);
    }
    params.push_back(&bias_);
    return params;
  }

  void zero_grad() override {
    for (auto& w : weights_) {
      w.grad = 0;
    }
    bias_.grad = 0;
  }

private:
  std::vector<Value> weights_;
  std::vector<Value> inputs_;
  Value bias_;
  bool nonlin_; 
};

class Layer : public Module {
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

  std::vector<Value*> parameters() override {
    std::vector<Value*> params;
    for (auto& neuron : neurons_) {
      auto neuron_params = neuron.parameters();
      params.insert(params.end(), neuron_params.begin(), neuron_params.end());
    }
    return params;
  }

  void zero_grad() override {
    for (auto& neuron : neurons_) {
      neuron.zero_grad();
    }
  }

private:
  std::vector<Neuron> neurons_;
  size_t nin_;
  bool nonlin_;
};

class MLP : public Module {
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

  std::vector<Value*> parameters() override {
    std::vector<Value*> params;
    for (auto& layer : layers_) {
      auto layer_params = layer.parameters();
      params.insert(params.end(), layer_params.begin(), layer_params.end());
    }
    return params;
  }

  void zero_grad() override {
    for (auto& layer : layers_) {
      layer.zero_grad();
    }
  }

private:
  std::vector<Layer> layers_;
};