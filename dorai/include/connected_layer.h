#pragma once
#include"image.h"

class ConnectedLayer {
public:
    ConnectedLayer(int inputs, int outputs);
    ~ConnectedLayer();
    void run(double* input);
    void backpropagate(double* input);
    void calculateUpdates(double* input);
    void update(double step);
    double* getOutput();
private:
    int _inputs;
    int _outputs;
    double* _weights;
    double* _biases;
    double* _weight_updates;
    double* _bias_updates;
    double* _output;
};