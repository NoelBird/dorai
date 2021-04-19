#include "activations.h"

#include <math.h>

double identityActivation(double x)
{
    return x;
}
double identityGradient(double x)
{
    return 1;
}

double reluActivation(double x)
{
    return x * (x > 0);
}
double reluGradient(double x)
{
    return (x >= 0);
}

double sigmoidActivation(double x)
{
    return 1. / (1. + exp(-x));
}

double sigmoidGradient(double x)
{
    return x * (1. - x);
}

