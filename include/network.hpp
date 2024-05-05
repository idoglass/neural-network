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
    using Target = vector<double>;

    Network(    string name_,
                vector<int> hidden_, 
                const IMath &math_, 
                Storage &storage_ = Storage(), 
                double bias = 1.0,
                int threads_ = 12);
    ~Network();

    void GetResult(Result &results_) const;
    void FeedForward(const Input &results_);
    void PropagateBack(const Target & targets_);

    double GetRecentAverageError(void) const { return m_recent_avg_err; }

    void PrintTopology() const;

private:
    vector<int> InitTopology(const vector<int> &hidden_layers_);
    void InitLayers(double bias_);

    const IMath &m_math;
    Storage &m_storage;
    vector<int> m_topology;
    ThreadPool m_thread_pool;
    const string m_name;
    vector<Layer> m_layers;
    double m_recent_avg_err;
    double m_error;
    static double m_recentAverageSmoothingFactor;
    
};
 
} // namespace neural_network

namespace neural_network
{
template<int INPUT, int OUTPUT>
double Network<INPUT, OUTPUT>::m_recentAverageSmoothingFactor = 100.0;

template<int INPUT, int OUTPUT>
Network<INPUT, OUTPUT>::Network(string name_,
                                vector<int> hidden_, 
                                const IMath &math_, 
                                Storage &storage_, 
                                double bias_,
                                int threads_) : 
                                        m_math(math_),
                                        m_storage(storage_),
                                        m_topology(InitTopology(hidden_)),
                                        m_thread_pool(),
                                        m_name(name_),
                                        m_layers(),
                                        m_recent_avg_err()
{
    std::cout << m_name << " network"<< std::endl;
    InitLayers(bias_);
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
void  Network<INPUT, OUTPUT>::GetResult(Result &result_) const
{
    const Result &output_layer = m_layers.back().GetValues();

    std::copy(output_layer.begin(), output_layer.end(), std::back_inserter(result_));
}

template<int INPUT, int OUTPUT>
void  Network<INPUT, OUTPUT>::FeedForward(const Input &input_)
{
    vector<Neuron> &input_layer = m_layers[0].GetNeurons();

    for (size_t i = 0; i < m_topology[0]; i++)
    {
        input_layer[i].SetValue(input_[i]);
    }
    for (size_t i = 1; i < m_topology.size(); i++)
    {
        m_layers[i].FeedForward(m_layers[i - 1]);
    }

}

template<int INPUT, int OUTPUT>
void Network<INPUT, OUTPUT>::InitLayers(double bias_)
{
    int i = 0;
    for (size_t size : m_topology)
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

        m_layers.push_back(Layer(size, bias_, m_math, type));
        
        if (i != 0)
        {
            m_layers[i].SetConnections(m_topology[i - 1]);
        }
       
        ++i;
    }
    }
    
template<int INPUT, int OUTPUT>
void Network<INPUT, OUTPUT>::PropagateBack(const Target &targets_)
{
    
    Layer &output_layer = m_layers.back();
    m_error = 0.0;

    for (unsigned n = 0; n < output_layer.Size(); ++n) {
        double delta = targets_[n] - output_layer[n].GetValue();
        m_error += delta * delta;
    }

    m_error /= output_layer.Size(); // get average error squared
    m_error = m_math.Sqrt(m_error); // RMS

    m_recent_avg_err =
            (m_recent_avg_err * m_recentAverageSmoothingFactor + m_error)
            / (m_recentAverageSmoothingFactor + 1.0);

    for (unsigned n = 0; n < output_layer.Size(); ++n) {
        output_layer[n].CalcOutputGradients(targets_[n]);
    }

    for (unsigned layerNum = m_layers.size() - 2; layerNum > 0; --layerNum) {
        Layer &hiddenLayer = m_layers[layerNum];
        Layer &nextLayer = m_layers[layerNum + 1];

        for (unsigned n = 0; n < hiddenLayer.Size(); ++n) {
            hiddenLayer[n].CalcHiddenGradients(nextLayer);
        }
    }

    for (unsigned layerNum = m_layers.size() - 1; layerNum > 0; --layerNum) {
        Layer &layer = m_layers[layerNum];
        Layer &prevLayer = m_layers[layerNum - 1];

        for (unsigned n = 0; n < layer.Size() ; ++n) {
   
            layer[n].UpdateInputWeights(prevLayer);
        }
    }
}



} // namespace neural_network

#endif //__NEURAL_NETWORK_NETWORK__