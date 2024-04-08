/*
network
*/


#ifndef __NEURAL_NETWORK_NETWORK__
#define __NEURAL_NETWORK_NETWORK__

#include <iostream>
#include <vector>

#include "i_math.hpp"
#include "thread_pool.hpp"
#include "storage.hpp"
#include "typedefs.hpp"
#include "layer.hpp"

namespace neural_network
{

class Topology
{
public:
    Topology(int size_ = 1, LayerType type_ = HIDDEN) : m_size(size_), m_type(type_) {};
    int m_size;
    LayerType m_type;
};
    
//TOPOLOGY must be of Topology class
template<int INPUT, int OUTPUT>
class Network
{
public:
    using Result = vector<double>;
    using Input = vector<double>;

    Network(    string name_,
                vector<int> hidden_, 
                const IMath &math_, 
                Storage &storage_ = Storage(), 
                int threads_ = 12);
    ~Network();

    void GetResult(Result &results_) const;
    void FeedForword(Input &results_) const;
    void PropogateBack(Input &results_) const;

    void PrintTopology() const;

private:
    vector<int> InitTopology(const vector<int> &hidden_layers_);
    vector<Layer> InitLayers();

    const IMath &m_math;
    Storage &m_storage;
    vector<int> m_topology;
    ThreadPool m_thread_pool;
    const string m_name;
    vector<Layer> m_layers;
    
};
 
} // namespace neural_network

namespace neural_network
{
template<int INPUT, int OUTPUT>
Network<INPUT, OUTPUT>::Network(string name_,
                                vector<int> hidden_, 
                                const IMath &math_, 
                                Storage &storage_, 
                                int threads_) : 
                                        m_math(math_),
                                        m_storage(storage_),
                                        m_topology(InitTopology(hidden_)),
                                        m_thread_pool(),
                                        m_name(name_),
                                        m_layers(InitLayers())
{
    std::cout << m_name << std::endl;
    std::cout << m_topology.size() << std::endl;
    PrintTopology();

}

template<int INPUT, int OUTPUT>
Network<INPUT, OUTPUT>::~Network()
{}

template<int INPUT, int OUTPUT>
void Network<INPUT, OUTPUT>::PrintTopology() const
{
    for (auto layer : m_layers)
    {
        std::cout << layer.Type() << " layer - " << layer.Size() << std::endl;
    }
    
}

template<int INPUT, int OUTPUT>
vector<int>  Network<INPUT, OUTPUT>::InitTopology(const vector<int> &hidden_) 
{
    vector<int> ret{INPUT};
    
    std::copy(hidden_.begin(), hidden_.end(), std::back_inserter(ret));

    ret.push_back(OUTPUT);

    return ret;
}

template<int INPUT, int OUTPUT>
vector<Layer>  Network<INPUT, OUTPUT>::InitLayers()
{
    vector<Layer> ret;
    int i = 0;
    for (int size : m_topology)
    {
        LayerType type = LayerType::HIDDEN;
        const int last_layer_index = m_topology.size() - 1;
        if (i == 0)
        {
            type = LayerType::INPUT;
        }
        else if (i == last_layer_index)
        {
            type = LayerType::OUTPUT;
        }

        ret.push_back(Layer(size, type));
        ++i;
    }
    
    return ret;
}
    
} // namespace neural_network


 #endif //__NEURAL_NETWORK_NETWORK__