#include "maxpool_layer.h"

void MaxpoolLayer::run(Image* input)
{
    int i, j, k;
    for (i = 0; i < _output->_h * _output->_w * _output->_c; ++i) _output->_data[i] = -DBL_MAX;
    for (i = 0; i < input->_h; ++i) {
        for (j = 0; j < input->_w; ++j) {
            for (k = 0; k < input->_c; ++k) {
                double val = input->getPixel(i, j, k);
                double cur = _output->getPixel(i / _stride, j / _stride, k);
                if (val > cur) _output->setPixel(i / _stride, j / _stride, k, val);
            }
        }
    }
}

Image* MaxpoolLayer::getOutput()
{
    return _output;
}
