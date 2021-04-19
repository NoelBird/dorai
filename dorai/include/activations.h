#pragma once

typedef enum {
    SIGMOID, RELU, IDENTITY
}ACTIVATOR_TYPE;

double reluActivation(double x);
double reluGradient(double x);
double sigmoidActivation(double x);
double sigmoidGradient(double x);
double identityActivation(double x);
double identityGradient(double x);
