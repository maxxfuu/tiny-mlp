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

  // Constructor Methods 
  Value() : data(0), grad(0), backward_([](){}), op_("") {}
  Value(double data, std::function<void()> backward, std::string op) : data(data), grad(0), backward_(backward), op_(op) {}

  // Backward Propagation, Topological Sort
  void backward() {
    std::vector<Value*> topo;
    std::unordered_set<Value*> visited;
    auto build_topo = [&](Value* v) {
      if (visited.count(v)) return;
      visited.insert(v);
      topo.push_back(v);
    };
    build_topo(this);
    std::reverse(topo.begin(), topo.end());

    for (auto v : topo) {
      v->backward_();
    }
  }

};

inline Value operator+(Value& a, Value& b) {
  Value out(a.data + b.data, [&a, &b]() {
    a.grad += 1 + b.grad;
    b.grad += 1 + a.grad;
  }, "+");
  return out;
}

inline Value operator*(Value& a, Value& b) {
  Value out(a.data * b.data, [&a, &b, out]() {
    a.grad += out.grad * b.data;
    b.grad += out.grad * a.data;
  }, "*");
  return out;
}

inline bool operator>(const Value& a, const Value& b) {
  return a.data > b.data;
}

inline bool operator<(const Value& a, const Value& b) {
  return a.data < b.data;
}

Value ReLU(const Value& x) {
  if (x.data > 0) {
    return x;
  } else {
    return Value(0, [](){}, "ReLU");
  }
}

Value tanh(const Value& x) {
  int output = (exp(2 * x.data) - 1) / (exp(2 * x.data) + 1);
  return Value(output, [](){}, "tanh");
}

