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

void ConvolutionalLayer::backpropagateLayer(Image* input)
{
    input->zero();
    for (int i = 0; i < _n; ++i) {
        input->backConvolve(_kernels[i], _stride, i, _output);
    }
}

void ConvolutionalLayer::backpropagateLayerConvolve(Image* input)
{
    int i, j;
    for (int i = 0; i < _n; ++i) {
        _kernels[i]->rotate();
    }

    input->zero();
    _output->upsample(_stride, _upsampled);
    for (int j = 0; j < input->getChannel(); ++j) {
        for (i = 0; i < _n; ++i) {
            _upsampled->twoDConvolve(i, _kernels[i], j, 1, input, j);
        }
    }

    for (int i = 0; i < _n; ++i) {
        _kernels[i]->rotate();
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