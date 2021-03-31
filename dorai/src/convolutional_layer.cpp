#include "convolutional_layer.h"

Image* ConvolutionalLayer::getKernel(int i)
{
    return &_kernels[i];
}

Image* ConvolutionalLayer::getOutput()
{
    return &_output;
}
