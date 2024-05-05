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

    int count = 0;
    vector<vector<double>> data({{1,1},{1,0},{0,1},{0,0}});
    vector<vector<double>> expected({{0},{1},{1},{0}});

    while(count < 20000)
    {
        int i = count % 4;
        network.FeedForward(data[i]);
        network.PropagateBack(expected[i]);         
        count++;
    }

    count = 0;
    while(count < 4)
    {
        int i = count % 4;
        network.FeedForward(data[i]);
        
        vector<double> results;
        network.GetResult(results);

        std::cout << "iteration num: " << count + 1 << std::endl;        
        std::cout << "data: " << data[i][0] << ", "<< data[i][1] << std::endl;        
        std::cout << "expected: " << expected[i][0] << std::endl;        
        std::cout << "results: " << (results[0] > 0.9 ? 1 : results[0] < 0.1 ? 0 : 2 )<< std::endl;              

        std::cout << std::endl;              

        count++;
    }
    std::cout << "avg error: " << network.GetRecentAverageError() << std::endl;              

    return 0;
}