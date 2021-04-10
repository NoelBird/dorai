#include"network.h"
#include "image.h"

#include "connected_layer.h"
#include "convolutional_layer.h"
#include "maxpool_layer.h"

Network::Network(int n): _n(n)
{
    _layers = new void* [_n];
    _types = new LAYER_TYPE[_n];
}

Network::~Network()
{
    delete[] _layers;
    delete[] _types;
}

void Network::run(Image* input)
{
    double* input_d = nullptr;
    for (int i = 0; i < _n; ++i) {
        if (_types[i] == LAYER_TYPE::CONVOLUTIONAL) {
            ConvolutionalLayer* layer = (ConvolutionalLayer*)_layers[i];
            layer->run(input);
            input = layer->getOutput();
            input_d = layer->getOutput()->getDataPointer();
        }
        else if (_types[i] == LAYER_TYPE::CONNECTED) {
            ConnectedLayer* layer = (ConnectedLayer*)_layers[i];
            layer->run(input_d);
            input_d = layer->getOutput();
        }
        else if (_types[i] == LAYER_TYPE::MAXPOOL) {
            MaxpoolLayer* layer = (MaxpoolLayer*)_layers[i];
            layer->run(input);
            input = layer->getOutput();
            input_d = layer->getOutput()->getDataPointer();
        }
    }
}

void Network::setTypes(int n, LAYER_TYPE t)
{
    _types[n] = t;
}

void Network::setLayers(int n, void* layer)
{
    _layers[n] = layer;
}

Image* Network::getImage()
{
    int i;
    for (i = _n - 1; i >= 0; --i) {
        if (_types[i] == LAYER_TYPE::CONVOLUTIONAL) {
            ConvolutionalLayer* layer = (ConvolutionalLayer*)_layers[i];
            return layer->getOutput();
        }
        else if (_types[i] == LAYER_TYPE::MAXPOOL) {
            MaxpoolLayer* layer = (MaxpoolLayer*)_layers[i];
            return layer->getOutput();
        }
    }
    
    Image* retn = new Image(1, 1, 1);
    return retn;
}

