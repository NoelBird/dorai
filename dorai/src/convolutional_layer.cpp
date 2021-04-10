#include "convolutional_layer.h"

double convolution_activation(double x);

void ConvolutionalLayer::run(Image* input)
{
    int i;
    for (i = 0; i < _n; ++i) {
        input->convolve(_kernels[i], _stride, i, _output);
    }
    for (i = 0; i < input->getHeight() * input->getWidth() * input->getChannel(); ++i) {
        input->setData(i, convolution_activation(input->getData(i)));
    }
}

Image* ConvolutionalLayer::getKernel(int i)
{
    return _kernels[i];
}

Image* ConvolutionalLayer::getOutput()
{
    return _output;
}

double convolution_activation(double x)
{
    return x * (x > 0);
}