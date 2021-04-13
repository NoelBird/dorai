#include "connected_layer.h"

#include <stdlib.h>
#include <string.h>

double activation(double x)
{
    return x * (x > 0);
}

double gradient(double x)
{
    return (x >= 0);
}

ConnectedLayer::ConnectedLayer(int inputs, int outputs): _inputs(inputs), _outputs(outputs)
{
    _output = new double[outputs];

    _weight_updates = new double[inputs * outputs];
    _weights = new double[inputs * outputs];
    for (int i = 0; i < inputs * outputs; ++i)
        _weights[i] = .5 - (double)rand() / RAND_MAX;

    _bias_updates = new double[outputs];
    _biases = new double[outputs];
    for (int i = 0; i < outputs; ++i)
        _biases[i] = (double)rand() / RAND_MAX;

    return;
}

ConnectedLayer::~ConnectedLayer()
{
    delete[] _output;
    delete[] _weight_updates;
    delete[] _weights;
    delete[] _bias_updates;
    delete[] _biases;
}

void ConnectedLayer::run(double* input)
{
    int i, j;
    for (i = 0; i < _outputs; ++i) {
        _output[i] = _biases[i];
        for (j = 0; j < _inputs; ++j) {
            _output[i] += input[j] * _weights[i * _outputs + j];
        }
        _output[i] = activation(_output[i]);
    }
}

void ConnectedLayer::backpropagate(double* input)
{
    int i, j;
    double* oldInput = new double[_inputs];
    memcpy(oldInput, input, _inputs * sizeof(double));
    memset(input, 0, _inputs * sizeof(double));

    for (i = 0; i < _outputs; ++i) {
        for (j = 0; j < _inputs; ++j) {
            input[j] += _output[i] * _weights[i * _outputs + j];
        }
    }
    for (j = 0; j < _inputs; ++j) {
        input[j] = input[j] * gradient(oldInput[j]);
    }
    delete[] oldInput;
}

void ConnectedLayer::calculateUpdates(double* input)
{
    int i, j;
    for (i = 0; i < _outputs; ++i) {
        _bias_updates[i] += _output[i];
        for (j = 0; j < _inputs; ++j) {
            _weight_updates[i * _outputs + j] += _output[i] * input[j];
        }
    }
}

void ConnectedLayer::update(double step)
{
    int i, j;
    for (i = 0; i < _outputs; ++i) {
        _biases[i] += step * _bias_updates[i];
        for (j = 0; j < _inputs; ++j) {
            int index = i * _outputs + j;
            _weights[index] = _weight_updates[index];
        }
    }
    memset(_bias_updates, 0, _outputs * sizeof(double));
    memset(_weight_updates, 0, _outputs * _inputs * sizeof(double));
}

double* ConnectedLayer::getOutput()
{
    return _output;
}

