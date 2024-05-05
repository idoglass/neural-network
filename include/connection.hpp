/*
Connection
*/

#ifndef __NEURAL_NETWORK_CONNECTION__
#define __NEURAL_NETWORK_CONNECTION__


namespace neural_network
{

struct Connection
{
public:
    Connection(double weight_) : m_weight(weight_), m_delta_weight(weight_) {};

    double GetWeight() const { return m_weight; };
    void SetWeight(double weight_ ) { m_weight = weight_; };

    double GetDeltaWeight() const { return m_delta_weight; };
    void SetDeltaWeight(double delta_weight_ ) { m_delta_weight = delta_weight_; };

private:
    double m_weight;
    double m_delta_weight;
};

} // namespace neural_network
#endif //__NEURAL_NETWORK_CONNECTION__