#include "convolutional_layer.h"

void ConvolutionalLayer::run(Image* input)
{
    int i;
    for (i = 0; i < _n; ++i) {
        input->convolve(_kernels[i], _stride, i, _output);
    }
    for (i = 0; i < input->getHeight() * input->getWidth() * input->getChannel(); ++i) {
        input->setData(i, convolutionActivation(input->getData(i)));
    }
}

void ConvolutionalLayer::learn(Image* input)
{
    int i;
    for (i = 0; i < _n; ++i) {
        input->kernelUpdate(_kernel_updates[i], _stride, i, _output);
    }
    Image* oldInput = new Image(*input);
    this->backpropagateLayerConvolve(input);
    for (i = 0; i < input->getHeight() * input->getWidth() * input->getChannel(); ++i) {
        input->setData(i, input->getData(i) * convolutionGradient(oldInput->getData(i)));
    }
    delete oldInput;
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

double convolutionActivation(double x)
{
    return x * (x > 0);
}

double convolutionGradient(double x)
{
    return (x >= 0);
}