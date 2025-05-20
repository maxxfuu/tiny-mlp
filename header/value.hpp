// Light weight scalar value class 
#include <cmath>

class Value {
public:
  double data;
  double grad;
  
  Value() : data(0) {}
  Value(double data) : data(data) {}

};

inline Value operator+(const Value& a, const Value& b) {
  return Value(a.data + b.data);
}

inline Value operator*(const Value& a, const Value& b) {
  return Value(a.data * b.data);
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
    return Value(0);
  }
}

Value tanh(const Value& x) {
  int output = (exp(2 * x.data) - 1) / (exp(2 * x.data) + 1);
  return Value(output);
}

