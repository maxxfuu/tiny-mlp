#pragma once
#include <vector>
#include "./value.hpp"

struct Value;

class Module {
public:
  virtual std::vector<Value*> parameters();
  virtual void zero_grad();
  virtual ~Module() = default;
};
