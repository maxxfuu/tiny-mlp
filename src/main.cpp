#include <iostream>
#include <vector>
#include <algorithm> // For std::max_element
#include <iomanip>   // For std::fixed and std::setprecision

#include "../header/engine.hpp"
#include "../header/nn.hpp"
#include "./mnist_utils.hpp" // Include the MNIST utilities

// Prediction function
int predict(MLP& network, const std::vector<Value>& input_image) {
    std::vector<Value> logits = network.forward_pass(input_image);
    std::vector<Value> probabilities = softmax(logits); // Assumes softmax is in engine.hpp scope

    // Find the index of the max probability
    auto max_it = std::max_element(probabilities.begin(), probabilities.end(), 
                                   [](const Value& a, const Value& b) {
                                       return a.data < b.data;
                                   });
    return std::distance(probabilities.begin(), max_it);
}

// Training function for a single batch/step (conceptual)
// This will be integrated into the main training loop.
// The core logic: zero_grad, forward, loss, backward, optimizer_step.

int main() {
    // MNIST specific parameters
    const int INPUT_SIZE = 28 * 28; // MNIST images are 28x28 pixels
    const int OUTPUT_SIZE = 10;     // 10 classes for digits 0-9

    // Load MNIST Dataset
    MNISTDataset dataset;
    if (!dataset.load("data")) { // Assuming data is in ./data relative to the executable
                                    // Or specify absolute path if needed.
        std::cerr << "Could not load MNIST dataset. Exiting." << std::endl;
        return 1;
    }

    // Define MLP architecture: e.g., 784 -> 128 -> 64 -> 10
    std::vector<size_t> architecture = {static_cast<size_t>(INPUT_SIZE), 128, 64, static_cast<size_t>(OUTPUT_SIZE)};
    MLP network(architecture);

    // Training parameters
    const int EPOCHS = 10;
    const int BATCH_SIZE = 32;
    double learning_rate = 0.001; // Reduced from 0.01 to 0.001

    std::cout << "Starting training..." << std::endl;

    for (int epoch = 0; epoch < EPOCHS; ++epoch) {
        float total_epoch_loss = 0.0f;
        int batches_processed = 0;

        // Shuffle training data (optional but recommended)
        // For simplicity, not implemented here, but consider shuffling dataset.train_data.images and dataset.train_labels together.

        for (int i = 0; i < dataset.train_data.num_images; i += BATCH_SIZE) {
            network.zero_grad(); // Zero gradients for all parameters in the network
            
            Value accumulated_batch_loss = Value(0.0, {}, [](){}, "batch_loss_acc");
            int actual_batch_size = 0;

            for (int j = 0; j < BATCH_SIZE && (i + j) < dataset.train_data.num_images; ++j) {
                actual_batch_size++;
                int image_idx = i + j;

                // Prepare input: convert vector<double> to vector<Value>
                std::vector<Value> input_values;
                input_values.reserve(INPUT_SIZE);
                for (int k = 0; k < INPUT_SIZE; ++k) {
                    // Assuming dataset.train_data.images[image_idx] is already a flat vector<double>
                    input_values.push_back(Value(dataset.train_data.images[image_idx][k], {}, [](){}, "input_pixel"));
                }

                // Forward pass
                std::vector<Value> logits = network.forward_pass(input_values);
                std::vector<Value> probabilities = softmax(logits);

                // Compute loss for this single sample
                int target_label = dataset.train_labels[image_idx];
                Value sample_loss = cross_entropy_loss(probabilities, target_label);
                accumulated_batch_loss = accumulated_batch_loss + sample_loss;
            }

            if (actual_batch_size == 0) continue;

            // Average loss over the batch
            Value batch_size_val(static_cast<double>(actual_batch_size), {}, [](){}, "batch_size_val");
            Value average_batch_loss = accumulated_batch_loss / batch_size_val; 
            total_epoch_loss += average_batch_loss.data;
            batches_processed++;

            // Backward pass (on the averaged batch loss)
            average_batch_loss.backward();

            // Update parameters (SGD)
            for (Value* param : network.parameters()) {
                param->data -= learning_rate * param->grad;
            }

            if ((batches_processed % 100) == 0) { // Print progress every 100 batches
                std::cout << "Epoch: " << epoch + 1 << "/" << EPOCHS 
                          << ", Batch: " << batches_processed 
                          << ", Avg Batch Loss: " << std::fixed << std::setprecision(4) << average_batch_loss.data 
                          << std::endl;
            }
        }

        float avg_epoch_loss = (batches_processed > 0) ? (total_epoch_loss / batches_processed) : 0.0f;
        std::cout << "Epoch: " << epoch + 1 << " completed. Average Epoch Loss: " << std::fixed << std::setprecision(4) << avg_epoch_loss << std::endl;

        // Evaluate on test set after each epoch
        int correct_predictions = 0;
        for (int i = 0; i < dataset.test_data.num_images; ++i) {
            std::vector<Value> test_image_values;
            test_image_values.reserve(INPUT_SIZE);
            for (int k = 0; k < INPUT_SIZE; ++k) {
                test_image_values.push_back(Value(dataset.test_data.images[i][k], {}, [](){}, "test_pixel"));
            }
            int predicted_label = predict(network, test_image_values);
            if (predicted_label == dataset.test_labels[i]) {
                correct_predictions++;
            }
        }
        double accuracy = static_cast<double>(correct_predictions) / dataset.test_data.num_images;
        std::cout << "Test Accuracy after Epoch " << epoch + 1 << ": " << std::fixed << std::setprecision(4) << (accuracy * 100.0) << "%" << std::endl;
    
        // Learning rate decay
        learning_rate *= 0.95; // Decay learning rate by 5% each epoch
    }

    std::cout << "Training finished." << std::endl;

    // Example of predicting a single image (e.g., first test image)
    if (dataset.test_data.num_images > 0) {
        std::vector<Value> single_test_image;
        single_test_image.reserve(INPUT_SIZE);
        for(int k=0; k < INPUT_SIZE; ++k) {
            single_test_image.push_back(Value(dataset.test_data.images[0][k], {}, [](){}, ""));
        }
        int final_prediction = predict(network, single_test_image);
        std::cout << "Prediction for the first test image: " << final_prediction 
                  << " | Actual label: " << static_cast<int>(dataset.test_labels[0]) << std::endl;
    }

    return 0;
} 