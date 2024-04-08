/*
layer
*/

#ifndef __NEURAL_NETWORK_LAYER__
#define __NEURAL_NETWORK_LAYER__

#include "neuron.hpp"
namespace neural_network
{
enum LayerType {INPUT, HIDDEN, OUTPUT};

class Layer
{
public:
    Layer(int size_, LayerType type_ = HIDDEN): m_neurons(size_), m_type(type_) {};

    size_t Size() { return m_neurons.size(); };
    const LayerType &Type() { return m_type; };

  friend inline std::ostream& operator<<(std::ostream& os_, const LayerType& type_);
private:
    vector<Neuron> m_neurons;
    LayerType m_type;
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