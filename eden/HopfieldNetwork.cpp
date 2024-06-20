#include "HopfieldNetwork.h"
# include "iostream"

HopfieldNetwork::HopfieldNetwork(int size) : size(size) {
    weights.resize(size, std::vector<int>(size, 0));
}

void HopfieldNetwork::train(const std::vector<std::vector<int>>& patterns) {
    for (const auto& pattern : patterns) {
        updateWeights(pattern);
    }
}

void HopfieldNetwork::updateWeights(const std::vector<int>& pattern) {
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            if (i != j) {
                weights[i][j] += pattern[i] * pattern[j];
            }
        }
    }
}

std::vector<int> HopfieldNetwork::recall(const std::vector<int>& pattern) {
    std::vector<int> state = pattern;
    std::vector<int> prevState;

    do {
        prevState = state;
        for (int i = 0; i < size; ++i) {
            int sum = 0;
            for (int j = 0; j < size; ++j) {
                sum += weights[i][j] * state[j];
            }
            state[i] = signum(sum);
        }
    } while (state != prevState);

    return state;
}

int HopfieldNetwork::signum(int x) {
    return (x >= 0) ? 1 : -1;
}

int HopfieldNetwork::calculateCapacity() {
    return 0.14*size;
}
