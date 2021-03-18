//#include "connected_layer.h"
//#include "convolutional_layer.h"
//#include "maxpool_layer.h"
//#include "network.h"
//#include "image.h"

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
    Image* image = new Image();
    image->loadFromFile("dog.jpg");
    //show_image(dog, "Test Backpropagate Input");
    //image dog_copy = copy_image(dog);
    //convolutional_layer cl = make_convolutional_layer(dog.h, dog.w, dog.c, n, size, stride);
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

int main()
{
    test_backpropagate();
    //test_convolve();
    //test_upsample();
    //test_rotate();
    //test_load();
    /*test_network();*/
    //test_convolutional_layer();
    //test_color();
    cv::waitKey(0);
    cv::waitKey(0);
    
    return 0;
}
