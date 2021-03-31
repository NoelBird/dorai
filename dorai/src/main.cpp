//#include "connected_layer.h"
//#include "network.h"
#include "image.h"
#include "convolutional_layer.h"
#include "maxpool_layer.h"

#include <time.h>
#include <stdlib.h>
#include <stdio.h>

#include "opencv2/opencv.hpp"
#include "image.h"

using namespace cv;

void test_backpropagate()
{

    int n = 3;
    int size = 4;
    int stride = 10;
    Image* dog = new Image();
    dog->loadFromFile("images/dog.jpg");
    dog->show("Test Backpropagate Input");
    Image dog_copy(*dog);
    // convolutional_layer cl = make_convolutional_layer(dog.h, dog.w, dog.c, n, size, stride);
    //run_convolutional_layer(dog, cl);
    //show_image(cl.output, "Test Backpropagate Output");
    //int i;
    //clock_t start = clock(), end;
    //for (i = 0; i < 100; ++i) {
    //    backpropagate_layer(dog_copy, cl);
    //}
    //end = clock();
    //printf("Backpropagate: %lf seconds\n", (double)(end - start) / CLOCKS_PER_SEC);
    //start = clock();
    //for (i = 0; i < 100; ++i) {
    //    backpropagate_layer_convolve(dog, cl);
    //}
    //end = clock();
    //printf("Backpropagate Using Convolutions: %lf seconds\n", (double)(end - start) / CLOCKS_PER_SEC);
    //show_image(dog_copy, "Test Backpropagate 1");
    //show_image(dog, "Test Backpropagate 2");
    //subtract_image(dog, dog_copy);
    //show_image(dog, "Test Backpropagate Difference");
}

void test_color() {
    Image dog;
    dog.loadFromFile("images/test_color.png");
    dog.showImageLayers("Test Color");
}

void test_load() {
    Image* dog = new Image();
    dog->loadFromFile("images/test_dog.jpg");
    dog->show("dog");
    dog->showImageLayers("dogLabels");
    delete dog;
}

void test_upsample() {
    Image* dog = new Image();

    dog->loadFromFile("images/dog.jpg");
    int n = 3;
    Image* up = new Image(n * dog->getHeight() , n * dog->getWidth(), dog->getChannel());
    dog->upsample(n, up);
    dog->show("Test Upsample");
    up->showImageLayers("Test Upsample");

    delete dog;
    delete up;
}

void test_rotate()
{
    int i;
    Image* dog = new Image();
    dog->loadFromFile("images/dog.jpg");
    clock_t start = clock(), end;
    for (i = 0; i < 1001; ++i) {
        dog->rotate();
    }
    end = clock();
    printf("Rotations: %lf seconds\n", (double)(end - start) / CLOCKS_PER_SEC);
    dog->show("Test Rotate");

    Image* random = new Image();
    random->makeRandomImage(3, 3, 3);
    random->show("Test Rotate Random");
    random->rotate();
    random->show("Test Rotate Random");
    random->rotate();
    random->show("Test Rotate Random");
}

void test_convolve()
{
    Image* dog = new Image();
    dog->loadFromFile("images/dog.jpg");
    dog->showImageLayers("Dog");
    printf("dog channels %d\n", dog->getChannel());
    Image* kernel = new Image();
    kernel->makeRandomImage(3, 3, dog->getChannel());
    Image* edge = new Image();
    edge->make(dog->getHeight(), dog->getWidth(), 1);
    clock_t start = clock(), end;
    for (int i = 0; i < 100; ++i) {
        dog->convolve(kernel, 1, 0, edge);
    }
    end = clock();
    printf("Convolutions: %lf seconds\n", (double)(end - start) / CLOCKS_PER_SEC);
    edge->show("Test Convolve");
}

void test_convolutional_layer()
{

    srand(0);
    Image* dog = new Image();
    dog->loadFromFile("images/test_dog.jpg");
    int n = 5;
    int stride = 1;
    int size = 8;
    /*ConvolutionalLayer layer(dog->getHeight(), dog->getWidth(), dog->getChannel(), n, size, stride);
    char buff[256];
    for (int i = 0; i < n; ++i) {
        sprintf_s(buff, "Kernel %d", i);
        layer.getKernel(i)->show(buff);
    }
    layer.run(dog);

    MaxpoolLayer mlayer(layer.getOutput()->getHeight(), layer.getOutput()->getWidth(), layer.getOutput()->getChannel(), 3);
    layer.run(layer.getOutput());

    mlayer.getOutput()->show("Test Maxpool Layer");*/
}

int main()
{
    //TODO
    //test_backpropagate();
    /*test_network();*/
    test_convolutional_layer();
    
    //DONE
    //test_load();
    //test_color();
    //test_upsample();
    //test_rotate();
    //test_convolve();
    cv::waitKey(0);
    
    return 0;
}
