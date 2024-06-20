#ifndef HOPFIELDNETWORK_H
#define HOPFIELDNETWORK_H

#include <vector>
#include <iostream>

class HopfieldNetwork {
public:
    HopfieldNetwork(int size);

    void train(const std::vector<std::vector<int>>& patterns);
    std::vector<int> recall(const std::vector<int>& pattern);
    int calculateCapacity();

private:
    int size;
    std::vector<std::vector<int>> weights;

    void updateWeights(const std::vector<int>& pattern);
    int signum(int x);
};

#endif // HOPFIELDNETWORK_H
