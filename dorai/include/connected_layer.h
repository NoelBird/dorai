#pragma once
#include"image.h"
#include "activations.h"

class ConnectedLayer {
public:
    ConnectedLayer(int inputs, int outputs, ACTIVATOR_TYPE activator);
    ~ConnectedLayer();
    void run(double* input);
    void learn(double* input);
    void update(double step);
    void backpropagate(double* input);
    void calculateUpdate(double* input);
    double* getOutput();
private:
    int _inputs;
    int _outputs;
    double* _weights;
    double* _biases;
    double* _weight_updates;
    double* _bias_updates;
    double* _output;
    double (*_activation)(double x);
    double (*_gradient)(double x);
};