#pragma once
#include "image.h"

class MaxpoolLayer;

class ConvolutionalLayer {
public:
    ConvolutionalLayer(int w, int h, int c, int n, int size, int stride): _n(n), _stride(stride) {
        _kernels = new Image*[n];
        _kernel_updates = new Image*[n];
        for (int i = 0; i < n; ++i) {
            _kernels[i] = new Image();
            _kernels[i]->makeRandomKernel(size, c);
            _kernel_updates[i] = new Image();
            _kernel_updates[i]->makeRandomKernel(size, c);
        }
        _output = new Image();
        _output->make((h - 1) / stride + 1, (w - 1) / stride + 1, n);
        _upsampled = new Image();
        _upsampled->make(h, w, n);
    };
    ~ConvolutionalLayer() {
        delete[]_kernels;
        delete[]_kernel_updates;
    };
    void run(Image* input);
    void backpropagateLayer(Image input);
    void backpropagateLayerConvolve(Image input);
    Image* getKernel(int i);
    Image* getOutput();
private:
    int _n;
    int _stride;
    Image** _kernels;
    Image** _kernel_updates;
    Image* _upsampled;
    Image* _output;

    friend class MaxpoolLayer;
};