#include "network.hpp"
#include "storage.hpp"

using namespace neural_network;

int main()
{
    std::vector<int> hidden_layers = {4};
    IMath *math = nullptr;
    Storage storage;

    Network<2,1> network(hidden_layers, math, storage);

    return 0;
}