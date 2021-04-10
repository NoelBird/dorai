#pragma once
#include "image.h"

class Image;

class MaxpoolLayer {
public:
    MaxpoolLayer(int h, int w, int c, int stride) : _stride(stride), _output(nullptr) {
        _output = new Image((h - 1) / stride + 1, (w - 1) / stride + 1, c);
        _output->make(h, w, c);
    };
    ~MaxpoolLayer() {
        delete _output;
    };
    void run(Image* input);
    Image* getOutput();
private:
    Image* _output;
    int _stride;
    
    
};

