#pragma once
#include "image.h"

enum class LAYER_TYPE {
    CONVOLUTIONAL,
    CONNECTED,
    MAXPOOL
};

class Network {
public:
    Network(int n);
    ~Network();
    void run(Image* input);
    void setTypes(int n, LAYER_TYPE t);
    void setLayers(int n, void* layer);
    Image* getImage();
private:
    int _n;
    void** _layers;
    LAYER_TYPE* _types;
};