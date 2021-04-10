//#include "connected_layer.h"
//#include "network.h"
#include "image.h"
#include "convolutional_layer.h"
#include "connected_layer.h"
#include "maxpool_layer.h"
#include "network.h"

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
    ConvolutionalLayer* cl = new ConvolutionalLayer(dog->getHeight(), dog->getWidth(), dog->getChannel(), n, size, stride);
    cl->run(dog);
    cl->getOutput()->show("Test Backpropagate Output");
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

    srand(time(0));
    Image* dog = new Image();
    dog->loadFromFile("images/test_dog.jpg");
    int n = 5;
    int stride = 1;
    int size = 8;
    ConvolutionalLayer* layer = new ConvolutionalLayer(dog->getHeight(), dog->getWidth(), dog->getChannel(), n, size, stride);

    char buff[256];
    for (int i = 0; i < n; ++i) {
        sprintf_s(buff, "Kernel %d", i);
        layer->getKernel(i)->show(buff);
    }
    layer->run(dog);

    MaxpoolLayer* mlayer = new MaxpoolLayer(layer->getOutput()->getHeight(), layer->getOutput()->getWidth(), layer->getOutput()->getChannel(), 3);
    mlayer->run(layer->getOutput());

    mlayer->getOutput()->showImageLayers("Test Maxpool Layer");
}

void test_network()
{
    Network* net = new Network(11);
    net->setTypes(0, LAYER_TYPE::CONVOLUTIONAL);
    net->setTypes(1, LAYER_TYPE::MAXPOOL);
    net->setTypes(2, LAYER_TYPE::CONVOLUTIONAL);
    net->setTypes(3, LAYER_TYPE::MAXPOOL);
    net->setTypes(4, LAYER_TYPE::CONVOLUTIONAL);
    net->setTypes(5, LAYER_TYPE::CONVOLUTIONAL);
    net->setTypes(6, LAYER_TYPE::CONVOLUTIONAL);
    net->setTypes(7, LAYER_TYPE::MAXPOOL);
    net->setTypes(8, LAYER_TYPE::CONNECTED);
    net->setTypes(9, LAYER_TYPE::CONNECTED);
    net->setTypes(10, LAYER_TYPE::CONNECTED);

    Image* dog = new Image();
    dog->loadFromFile("images/test_hinton.jpg");

    int n = 48;
    int stride = 4;
    int size = 11;
    ConvolutionalLayer* cl = new ConvolutionalLayer(dog->getHeight(), dog->getWidth(), dog->getChannel(), n, size, stride);
    MaxpoolLayer* ml = new MaxpoolLayer(cl->getOutput()->getHeight(), cl->getOutput()->getWidth(), cl->getOutput()->getChannel(), 2);

    n = 128;
    size = 5;
    stride = 1;
    ConvolutionalLayer* cl2 = new ConvolutionalLayer(ml->getOutput()->getHeight(), ml->getOutput()->getWidth(), ml->getOutput()->getChannel(), n, size, stride);
    MaxpoolLayer* ml2 = new MaxpoolLayer(cl2->getOutput()->getHeight(), cl2->getOutput()->getWidth(), cl2->getOutput()->getChannel(), 2);

    n = 192;
    size = 3;
    ConvolutionalLayer* cl3 = new ConvolutionalLayer(ml2->getOutput()->getHeight(), ml2->getOutput()->getWidth(), ml2->getOutput()->getChannel(), n, size, stride);
    ConvolutionalLayer* cl4 = new ConvolutionalLayer(cl3->getOutput()->getHeight(), cl3->getOutput()->getWidth(), cl3->getOutput()->getChannel(), n, size, stride);
    n = 128;
    ConvolutionalLayer* cl5 = new ConvolutionalLayer(cl4->getOutput()->getHeight(), cl4->getOutput()->getWidth(), cl4->getOutput()->getChannel(), n, size, stride);
    MaxpoolLayer* ml3 = new MaxpoolLayer(cl5->getOutput()->getHeight(), cl5->getOutput()->getWidth(), cl5->getOutput()->getChannel(), 4);
    ConnectedLayer* nl = new ConnectedLayer(ml3->getOutput()->getHeight() * ml3->getOutput()->getWidth() * ml3->getOutput()->getChannel(), 4096);
    ConnectedLayer* nl2 = new ConnectedLayer(4096, 4096);
    ConnectedLayer* nl3 = new ConnectedLayer(4096, 1000);

    net->setLayers(0, cl);
    net->setLayers(1, ml);
    net->setLayers(2, cl2);
    net->setLayers(3, ml2);
    net->setLayers(4, cl3);
    net->setLayers(5, cl4);
    net->setLayers(6, cl5);
    net->setLayers(7, ml3);
    net->setLayers(8, nl);
    net->setLayers(9, nl2);
    net->setLayers(10, nl3);

    clock_t start = clock(), end;
    for (int i = 0; i < 10; ++i) {
        net->run(dog);
        dog->rotate();
    }
    end = clock();
    printf("Ran %lf second per iteration\n", (double)(end - start) / CLOCKS_PER_SEC / 10);

    net->getImage()->showImageLayers("Test Network Layer");
}

int main()
{
    //TODO
    //test_backpropagate();
    test_network();
    
    //DONE
    //test_load();
    //test_color();
    //test_upsample();
    //test_rotate();
    //test_convolve();
    //test_convolutional_layer();
    cv::waitKey(0);
    
    return 0;
}
