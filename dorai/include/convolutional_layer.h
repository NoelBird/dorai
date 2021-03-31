#pragma once
#include "image.h"

class MaxpoolLayer;

class ConvolutionalLayer {
public:
    ConvolutionalLayer(int w, int h, int c, int n, int size, int stride);
    ~ConvolutionalLayer();
    void run(const Image* input);
    void backpropagateLayer(Image input);
    void backpropagateLayerConvolve(Image input);
    Image* getKernel(int i);
    Image* getOutput();
private:
    int _n;
    int _stride;
    Image* _kernels;
    Image* _kernel_updates;
    Image _upsampled;
    Image _output;

    friend class MaxpoolLayer;
};