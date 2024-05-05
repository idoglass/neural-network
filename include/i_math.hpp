/*
storage
*/

#ifndef __NEURAL_NETWORK_I_MATH__
#define __NEURAL_NETWORK_I_MATH__

namespace neural_network
{

    class IMath
    {
    public:
        virtual double Rand() const = 0;
        virtual double Sqrt(double num_) const = 0;
        virtual double TransferFunction(double x) const = 0;
        virtual double TransferFunctionDerivative(double x) const = 0;


        
    };

} // namespace neural_network
#endif //__NEURAL_NETWORK_I_MATH__