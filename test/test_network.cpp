#include "network.hpp"
#include "storage.hpp"
#include "simple_math.hpp"
#include "typedefs.hpp"
using namespace neural_network;


int main()
{
    vector<int> hidden_layers = {4};
    SimpleMath math;
    Storage storage;

    Network<2,1> network("net1", hidden_layers, math, storage);
    
    return 0;
}