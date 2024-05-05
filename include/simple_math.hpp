/*
storage
*/


#ifndef __NEURAL_NETWORK_SIMPLE_MATH__
#define __NEURAL_NETWORK_SIMPLE_MATH__

#include <stdlib.h>     /* srand, rand */
#include <cmath>        /* tanh sqrt*/
#include <time.h>       /* time */

#include "i_math.hpp"

namespace neural_network
{
    
class SimpleMath : public IMath
{
public:
    SimpleMath() : IMath() 
    {
        std::srand(time(0));
    }

    virtual double Rand() const;
    virtual double Sqrt(double num_) const;
    virtual double TransferFunction(double x) const;
    virtual double TransferFunctionDerivative(double x) const;

};

inline double SimpleMath::Rand() const
{

    int num = rand() % 1000;

    return num / 1000.0;
}

inline double SimpleMath::Sqrt(double num_) const
{
    return std::sqrt(num_);
}

inline double SimpleMath::TransferFunction(double x) const
{
    return tanh(x);
}

inline double SimpleMath::TransferFunctionDerivative(double x) const
{
    return 1.0 - x * x;
}


 
} // namespace neural_network
 #endif //__NEURAL_NETWORK_SIMPLE_MATH__