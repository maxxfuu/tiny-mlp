#pragma once

#include <cmath>
#include <functional>
#include <unordered_set>
#include <vector>
#include <algorithm>

class Value {
public:
  double data;
  double grad;
  std::function<void()> backward_;
  std::string op_;
  std::vector<Value*> prev_;

  // Constructor Methods 
  Value() : data(0), grad(0), backward_([](){}), op_(""), prev_() {}
  Value(double data,
        std::vector<Value*> prev,
        std::function<void()> backward,
        std::string op)
      : data(data), grad(0), backward_(backward), op_(op), prev_(std::move(prev)) {}

  // Backward Propagation, Topological Sort
  void backward() {
    // Seed the output gradient
    this->grad = 1.0;

    // Build a topological ordering of the graph
    std::vector<Value*> topo;
    std::unordered_set<Value*> visited;

    std::function<void(Value*)> build_topo = [&](Value* v) {
      if (visited.count(v)) return;
      visited.insert(v);
      for (auto parent : v->prev_) build_topo(parent);
      topo.push_back(v);
    };
    build_topo(this);
    std::reverse(topo.begin(), topo.end());

    // Propagate gradients
    for (auto v : topo) v->backward_();
  }
};

inline Value operator+(Value& a, Value& b) {
  Value out(
      a.data + b.data,
      {&a, &b},
      [&a, &b, &out]() {
        a.grad += out.grad;
        b.grad += out.grad;
      },
      "+");
  return out;
}

inline Value operator+(Value& a, double b) {
  Value out(a.data + b, {&a}, [&a, &out]() {
    a.grad += 1 * out.grad;
  }, "+");
  return out;
}

inline Value operator+(double a, Value& b) {
  Value out(a + b.data, {&b}, [&b, &out]() {
    b.grad += 1 * out.grad;
  }, "+");
  return out;
}

inline Value operator*(Value& a, Value& b) {
  Value out(
      a.data * b.data,
      {&a, &b},
      [&a, &b, &out]() {
        a.grad += out.grad * b.data;
        b.grad += out.grad * a.data;
      },
      "*");
  return out;
}

inline Value operator*(Value& a, double b) {
  Value out(a.data * b, {&a}, [&a, b, &out]() {
    a.grad += b * out.grad;
  }, "*");
  return out;
}

inline Value operator*(double a, Value& b) {
  Value out(a * b.data, {&b}, [a, &b, &out]() {
    b.grad += a * out.grad;
  }, "*");
  return out;
}

inline Value operator-(Value& a, Value& b) {
  Value out(
    a.data - b.data,
    {&a, &b},
    [&a, &b, &out]() {
      a.grad += out.grad;
      b.grad -= out.grad;
    },
    "-");
  return out;
}

inline Value operator-(Value& a, double b_const) {
    Value out(a.data - b_const, {&a}, 
              [&a, &out]() {
                a.grad += out.grad; // d(out)/da = 1
              }, "-");
    return out;
}

inline Value operator-(double a_const, Value& b) {
    Value out(a_const - b.data, {&b}, 
              [&b, &out]() {
                b.grad -= out.grad; // d(out)/db = -1
              }, "-");
    return out;
}

inline Value operator/(Value& a, Value& b) {
    // Ensure b.data is not zero to prevent division by zero.
    // A more robust implementation might throw an exception or handle this case differently.
    if (b.data == 0) {
        // Handle division by zero, e.g., by returning a Value with NaN or infinity,
        // and possibly setting a very large gradient for b.
        // For simplicity, this example doesn't fully handle it but it's critical in a real scenario.
        return Value(std::numeric_limits<double>::quiet_NaN(), {&a, &b}, [](){}, "/");
    }
    Value out(
        a.data / b.data,
        {&a, &b},
        [&a, &b, &out]() {
            a.grad += out.grad / b.data;
            b.grad += out.grad * (-a.data / (b.data * b.data));
        },
        "/");
    return out;
}

inline Value operator/(Value& a, double b) {
    if (b == 0) {
        return Value(std::numeric_limits<double>::quiet_NaN(), {&a}, [](){}, "/");
    }
    Value out(a.data / b, {&a}, [&a, b, &out]() {
        a.grad += (1 / b) * out.grad;
    }, "/");
    return out;
}

inline Value operator/(double a, Value& b) {
    if (b.data == 0) {
        return Value(std::numeric_limits<double>::quiet_NaN(), {&b}, [](){}, "/");
    }
    Value out(a / b.data, {&b}, [a, &b, &out]() {
        b.grad += (-a / (b.data * b.data)) * out.grad;
    }, "/");
    return out;
}

inline bool operator>(const Value& a, const Value& b) {
  return a.data > b.data;
}

inline bool operator<(const Value& a, const Value& b) {
  return a.data < b.data;
}

Value ReLU(Value& x) {
  double output = x.data > 0 ? x.data : 0;
  Value out(
      output,
      {&x},
      [&x, &out]() {
        x.grad += (x.data > 0 ? 1.0 : 0.0) * out.grad;
      },
      "ReLU");
  return out;
}

Value tanh(Value& x) {
  double output = (std::exp(2 * x.data) - 1) / (std::exp(2 * x.data) + 1);
  Value out(
      output,
      {&x},
      [&x, &out]() {
        x.grad += (1 - out.data * out.data) * out.grad;
      },
      "tanh");
  return out;
}

Value exp(Value& a) {
    double val = std::exp(a.data);
    Value out(
        val,
        {&a},
        [&a, val, &out]() { 
            a.grad += val * out.grad; 
        },
        "exp");
    return out;
}

Value log(Value& a) {
    // Ensure a.data is positive for log
    if (a.data <= 0) {
        // Handle log of non-positive number
        return Value(std::numeric_limits<double>::quiet_NaN(), {&a}, [](){}, "log");
    }
    Value out(
        std::log(a.data),
        {&a},
        [&a, &out]() {
            a.grad += (1 / a.data) * out.grad;
        },
        "log");
    return out;
}

Value pow(Value& a, double p) {
    Value out(
        std::pow(a.data, p),
        {&a},
        [&a, p, &out]() {
            a.grad += (p * std::pow(a.data, p - 1)) * out.grad;
        },
        "pow");
    return out;
}

// Loss Functions
Value MSE(Value& y, Value& y_hat) {
  Value diff = y - y_hat;
  Value out = diff * diff; // Using overloaded operators
  out.op_ = "MSE";

  return out;
}

// Softmax and Cross-Entropy Loss for multi-class classification
std::vector<Value> softmax(std::vector<Value>& logits) {
    std::vector<Value> exps;
    exps.reserve(logits.size());
    double max_logit_val = logits[0].data;
    for (size_t i = 1; i < logits.size(); ++i) {
        if (logits[i].data > max_logit_val) {
            max_logit_val = logits[i].data;
        }
    }

    Value sum_exp_val = Value(0.0, {}, [](){}, "");
    for (auto& logit : logits) {
        // Create a temporary Value for (logit - max_logit_val)
        Value adjusted_logit = logit - max_logit_val;
        Value exp_val = exp(adjusted_logit);
        exps.push_back(exp_val);
        sum_exp_val = sum_exp_val + exp_val;
    }

    std::vector<Value> probs;
    probs.reserve(logits.size());
    for (auto& exp_val : exps) {
        probs.push_back(exp_val / sum_exp_val);
    }
    return probs;
}

Value cross_entropy_loss(std::vector<Value>& probs, int target_index) {
    // Ensure target_index is valid
    if (target_index < 0 || static_cast<size_t>(target_index) >= probs.size()) {
        return Value(0, {}, [](){}, "error_cross_entropy_invalid_index");
    }
    
    Value prob_target = probs[static_cast<size_t>(target_index)];
    double epsilon = 1e-12; 
    if (prob_target.data < epsilon) { 
    }

    Value log_prob = log(prob_target);
    Value neg_one = Value(-1.0, {}, [](){}, "const_neg_one");
    Value loss = neg_one * log_prob;
    loss.op_ = "cross_entropy";
    return loss;
}