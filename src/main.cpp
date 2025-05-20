#include <iostream>
#include "../header/nn.hpp"

int main() {
    // Create a simple neural network with architecture: 2 -> 3 -> 1
    std::vector<size_t> architecture = {2, 3, 3, 1};
    MLP network(architecture);

    // Create input values
    std::vector<Value> inputs = {Value(1.0, [](){}, ""), Value(0.5, [](){}, "")};

    // Perform forward pass
    std::vector<Value> output = network.forward_pass(inputs);

    // Print the output
    std::cout << "Network output: " << output[-1].data << "%" << std::endl;

    return 0;
} 