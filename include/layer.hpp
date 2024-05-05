/*
layer
*/

#ifndef __NEURAL_NETWORK_LAYER__
#define __NEURAL_NETWORK_LAYER__

#include "neuron.hpp"
#include "i_math.hpp"
#include "typedefs.hpp"

namespace neural_network
{
enum LayerType {INPUT, HIDDEN, OUTPUT};

class Layer
{
using Values = vector<vector<double>>;
public:
    Layer(size_t size_, double bias_, const IMath &math_, LayerType type_ = HIDDEN):  
                                        m_type(type_),
                                        m_neurons(InitNeurons(size_, bias_, math_)), 
                                        m_math(math_) 
    {};

    size_t Size() const { return m_neurons.size(); };
    const LayerType &Type() { return m_type; };

    Neuron &operator[](size_t i_) { return m_neurons[i_]; };
    const Neuron &At(size_t i_) const { return m_neurons[i_]; };
    vector<Neuron>& GetNeurons() { return m_neurons; };

    vector<Neuron> InitNeurons(size_t size_, double bias_, const IMath &math_)
    {
        vector<Neuron> res;

        for (size_t i = 0; i < size_; i++)
        {
            res.push_back({math_, bias_, i});
        }
        return res;
    };


    void FeedForward(const Layer &last_layer_)
    {
        for(Neuron &neuron : m_neurons)
        {
            neuron.CalcValue(last_layer_.GetValues());
        }
    }

    void SetConnections(size_t count_, const Values &values = Values())
    { 
        int i = 0;

        for(Neuron &neuron : m_neurons)
        {
            if(values.size() > 0)
            {
                neuron.SetConnections(count_, values[i]);   
            }
            else
            {
                neuron.SetConnections(count_);
            }
            ++i;
        }
    }

    vector<double> GetValues() const
    {
        vector<double> ret;
        for(const Neuron neuron : m_neurons)
        {
            ret.push_back(neuron.GetValue());
        }
        return ret;
    }

  friend inline std::ostream& operator<<(std::ostream& os_, const LayerType& type_);
private:
    LayerType m_type;
    vector<Neuron> m_neurons;
    const IMath &m_math;
};

inline std::ostream& operator<<(std::ostream& os_, const LayerType& type_)
{
    string str = "hidden";
    if(type_ == LayerType::INPUT)
    {
        str = "input";
    }
    else if(type_ == LayerType::OUTPUT)
    {
        str = "output";
    }
    os_ << str;
    return os_;
}
} // namespace neural_network
#endif //__NEURAL_NETWORK_LAYER__