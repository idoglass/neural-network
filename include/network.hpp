/*
network
*/


#ifndef __NEURAL_NETWORK_NETWORK__
#define __NEURAL_NETWORK_NETWORK__


#include <vector>

#include "i_math.hpp"
#include "thread_pool.hpp"
#include "storage.hpp"

namespace neural_network
{
    
//TOPOLOGY must be of Topology class
template<int INPUT, int OUTPUT>
class Network
{
public:
    using Result = std::vector<double>;
    using Input = std::vector<double>;

    Network(std::vector<int> hidden_, IMath *math_ = nullptr, Storage &storage_ = Storage(), int threads_ = 12);
    ~Network();

    void GetResult(Result &results_) const;
    void FeedForword(Input &results_) const;
    void PropogateBack(Input &results_) const;

private:
    IMath * m_math;
    Storage &m_storage;
    std::vector<int> m_topology;
    ThreadPool m_thread_pool;
};
 
} // namespace neural_network
 #endif //__NEURAL_NETWORK_NETWORK__