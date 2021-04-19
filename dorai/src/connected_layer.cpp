#include "connected_layer.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

ConnectedLayer::ConnectedLayer(int inputs, int outputs, ACTIVATOR_TYPE activator): _inputs(inputs), _outputs(outputs)
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

    if (activator == SIGMOID) {
        _activation = sigmoidActivation;
        _gradient = sigmoidGradient;
    }
    else if (activator == RELU) {
        _activation = reluActivation;
        _gradient = reluGradient;
    }
    else if (activator == IDENTITY) {
        _activation = identityActivation;
        _gradient = identityGradient;
    }


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
            _output[i] += input[j] * _weights[i * _inputs + j];
        }
        _output[i] = _activation(_output[i]);
    }
}

void ConnectedLayer::learn(double* input)
{
    calculateUpdate(input);
    backpropagate(input);
}

void ConnectedLayer::backpropagate(double* input)
{
    int i, j;
    for (j = 0; j < _inputs; ++j) {
        double grad = _gradient(input[j]);
        input[j] = 0;
        for (i = 0; i < _outputs; ++i) {
            input[j] += _output[i] * _weights[i * _inputs + j];
        }
        input[j] *= grad;
    }
}

void ConnectedLayer::calculateUpdate(double* input)
{
    int i, j;
    for (i = 0; i < _outputs; ++i) {
        _bias_updates[i] += _output[i];
        for (j = 0; j < _inputs; ++j) {
            _weight_updates[i * _inputs + j] += _output[i] * input[j];
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
    memset(_weight_updates, 0, _outputs * _inputs * sizeof(double)); // overflow can occur here.
}

double* ConnectedLayer::getOutput()
{
    return _output;
}

