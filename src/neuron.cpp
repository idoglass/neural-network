#include "neuron.hpp"
#include "layer.hpp"
#include "typedefs.hpp"
#include <iostream>


namespace neural_network
{
using Values = vector<double>;

Neuron::Neuron(const IMath &math_, double bias_, size_t my_index_) : 
                                        m_math(math_),
                                        m_value(0),
                                        m_bias(bias_),
                                        m_gradient(),
                                        m_my_index(my_index_),
                                        m_connections()
{
};

double Neuron::GetValue() const
{
    return m_value;
}

void Neuron::SetConnections(size_t count_, const Values &values)
{
    for (size_t i = 0; i < count_; i++)
    {
        if(values.size() == 0)
        {
            m_connections.push_back(Connection(m_math.Rand()));
        }
        else
        {
            m_connections.push_back(Connection(values[i]));
        }
    }
}

void Neuron::CalcValue(const vector<double> &prev_layer_values_)
{
    double sum = m_bias;

    for(int i = 0 ; i < prev_layer_values_.size(); ++i)
    {
        sum += prev_layer_values_[i] * m_connections[i].GetWeight();
    }

    m_value = m_math.TransferFunction(sum);
}

void Neuron::SetValue(double value_)
{
    m_value = value_;
}

void Neuron::CalcOutputGradients(double target_value_)
{
    double delta = target_value_ - m_value;
    m_gradient = delta * m_math.TransferFunctionDerivative(m_value);
}

void Neuron::CalcHiddenGradients(const Layer &next_layer_)
{
    double dow = 0.0;

    for (unsigned n = 0; n < next_layer_.Size(); ++n) {
        dow += next_layer_.At(n).m_connections[m_my_index].GetWeight() 
                                            * next_layer_.At(n).m_gradient;;
    }

    m_gradient = dow * m_math.TransferFunctionDerivative(m_value);
}

void Neuron::UpdateInputWeights(Layer &prev_layer_)
{
    double eta = 0.15; //net learning rate
    double alpha = 0.5; //momentum,

    for (unsigned n = 0; n < m_connections.size(); ++n) 
    {
        Connection &connection = m_connections[n];
        Neuron &neuron = prev_layer_[n];
        double oldDeltaWeight = connection.GetDeltaWeight();

        double newDeltaWeight = eta 
                                * neuron.GetValue() 
                                * m_gradient 
                                + alpha 
                                * oldDeltaWeight;

        connection.SetDeltaWeight(newDeltaWeight);
        connection.SetWeight(
            connection.GetWeight() + newDeltaWeight
        );
    }
}

} // namespace neural_network
