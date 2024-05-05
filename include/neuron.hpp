/*
The Neuron class represents a fundamental unit in an artificial neural network. 
It receives input from other neurons, performs calculations on that input, and 
outputs a value that can be used by subsequent layers in the network.

Public Methods:
    Constructor: Initializes a new Neuron object with the provided IMath 
        reference, bias value, and index.
    GetValue():Returns the current value (activation) of the neuron.
    **SetConnections(size_t count_, const Values &values = Values())`
        Establishes connections to a specified number of other neurons. 
        Optionally, provides initial weight values for the connections.
    **CalcValue(const vector<double> &prev_layer_values_)`
        Calculates the output value of the neuron based on the weighted sum of 
        inputs from the previous layer and applies the activation function.
    SetValue(double value_)
        Sets the current value (activation) of the neuron (primarily for 
        testing purposes).
    CalcOutputGradients(double target_value_)
        Calculates the output gradient (error signal) for the neuron in the 
        context of the given target value (used in the output layer).
    CalcHiddenGradients(const Layer &next_layer_)
        Calculates the hidden gradient (error signal) for the neuron in the 
        context of the next layer's gradients (used in hidden layers).
    UpdateInputWeights(Layer &prev_layer_)
        Updates the weights of the connections based on the learning rate and 
        the gradients of the neuron and the previous layer.
*/

#ifndef __NEURAL_NETWORK_NEURON__
#define __NEURAL_NETWORK_NEURON__

#include "connection.hpp"
#include "i_math.hpp"
#include "typedefs.hpp"

namespace neural_network
{
    class Layer;

    class Neuron
    {
    using Values = vector<double>;

    public:
        Neuron(const IMath &math_, double bias_, size_t my_index_);

        double GetValue() const;

        void SetConnections(size_t count_, const Values &values = Values());

        void CalcValue(const vector<double> &prev_layer_values_);

        void SetValue(double value_);

        void CalcOutputGradients(double target_value_);

        void CalcHiddenGradients(const Layer &next_layer_);

        void UpdateInputWeights(Layer &prev_layer_);

    private:
        const IMath &m_math; //A reference to an IMath object providing basic mathematical operations (addition, multiplication, etc.).
        double m_value; //The current value (activation) of the neuron.
        double m_bias; //The bias term added to the weighted sum of inputs before applying the activation function.
        unsigned m_my_index; //The index of this neuron within its layer.
        double m_gradient; //The current gradient (error signal) used for backpropagation.
        vector<Connection> m_connections; //A list of connections representing incoming connections from other neurons, each containing weight and delta weight values.
    };

} // namespace neural_network
#endif //__NEURAL_NETWORK_NEURON__