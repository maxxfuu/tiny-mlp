#pragma once

#include <fstream>
#include <vector>
#include <string>
#include <iostream> // For error reporting
#include <algorithm> // For std::reverse for endian conversion

// Function to reverse integer for MNIST file format (big-endian to little-endian)
int reverseInt(int i) {
    unsigned char c1, c2, c3, c4;
    c1 = i & 255;
    c2 = (i >> 8) & 255;
    c3 = (i >> 16) & 255;
    c4 = (i >> 24) & 255;
    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}

struct MNISTData {
    std::vector<std::vector<double>> images; // Flattened images
    std::vector<unsigned char> labels;
    int num_images;
    int img_rows;
    int img_cols;
};

MNISTData load_mnist_images(const std::string& image_file_path) {
    MNISTData data;
    std::ifstream file(image_file_path, std::ios::binary);

    if (file.is_open()) {
        int magic_number = 0;
        int number_of_images = 0;
        int n_rows = 0;
        int n_cols = 0;

        file.read((char*)&magic_number, sizeof(magic_number));
        magic_number = reverseInt(magic_number);
        if (magic_number != 2051) {
            std::cerr << "Invalid MNIST image file: Incorrect magic number " << magic_number << " in " << image_file_path << std::endl;
            return data; // Return empty data
        }

        file.read((char*)&number_of_images, sizeof(number_of_images));
        number_of_images = reverseInt(number_of_images);
        data.num_images = number_of_images;

        file.read((char*)&n_rows, sizeof(n_rows));
        n_rows = reverseInt(n_rows);
        data.img_rows = n_rows;

        file.read((char*)&n_cols, sizeof(n_cols));
        n_cols = reverseInt(n_cols);
        data.img_cols = n_cols;

        data.images.resize(number_of_images, std::vector<double>(n_rows * n_cols));

        for (int i = 0; i < number_of_images; ++i) {
            for (int r = 0; r < n_rows; ++r) {
                for (int c = 0; c < n_cols; ++c) {
                    unsigned char pixel = 0;
                    file.read((char*)&pixel, sizeof(pixel));
                    // Standardize to approximately mean 0 and std 1
                    // (pixel / 255.0 - 0.5) * 2
                    data.images[i][r * n_cols + c] = ((double)pixel / 127.5) - 1.0;
                }
            }
        }
        file.close();
    } else {
        std::cerr << "Cannot open image file: " << image_file_path << std::endl;
    }
    return data;
}

std::vector<unsigned char> load_mnist_labels(const std::string& label_file_path) {
    std::vector<unsigned char> labels;
    std::ifstream file(label_file_path, std::ios::binary);

    if (file.is_open()) {
        int magic_number = 0;
        int number_of_items = 0;

        file.read((char*)&magic_number, sizeof(magic_number));
        magic_number = reverseInt(magic_number);

        if (magic_number != 2049) {
            std::cerr << "Invalid MNIST label file: Incorrect magic number " << magic_number << " in " << label_file_path << std::endl;
            return labels; // Return empty labels
        }

        file.read((char*)&number_of_items, sizeof(number_of_items));
        number_of_items = reverseInt(number_of_items);
        labels.resize(number_of_items);

        for (int i = 0; i < number_of_items; ++i) {
            file.read((char*)&labels[i], sizeof(labels[i]));
        }
        file.close();
    } else {
        std::cerr << "Cannot open label file: " << label_file_path << std::endl;
    }
    return labels;
}

struct MNISTDataset {
    MNISTData train_data;
    std::vector<unsigned char> train_labels;
    MNISTData test_data;
    std::vector<unsigned char> test_labels;

    bool load(const std::string& data_path) {
        train_data = load_mnist_images(data_path + "/train-images-idx3-ubyte");
        train_labels = load_mnist_labels(data_path + "/train-labels-idx1-ubyte");
        test_data = load_mnist_images(data_path + "/t10k-images-idx3-ubyte");
        test_labels = load_mnist_labels(data_path + "/t10k-labels-idx1-ubyte");

        if (train_data.images.empty() || train_labels.empty() || test_data.images.empty() || test_labels.empty()) {
            std::cerr << "Failed to load one or more MNIST files." << std::endl;
            return false;
        }
        if (train_data.num_images != static_cast<int>(train_labels.size())) {
            std::cerr << "Train images and labels count mismatch." << std::endl;
            return false;
        }
        if (test_data.num_images != static_cast<int>(test_labels.size())) {
            std::cerr << "Test images and labels count mismatch." << std::endl;
            return false;
        }
        std::cout << "MNIST dataset loaded successfully." << std::endl;
        std::cout << "Training images: " << train_data.num_images << std::endl;
        std::cout << "Test images: " << test_data.num_images << std::endl;
        return true;
    }
}; 