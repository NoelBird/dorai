#pragma once

#include "opencv2/highgui/highgui_c.h"
#include <string>
#include "opencv2/imgproc/imgproc_c.h"

class Image {
public:
    Image(int h, int w, int c) : _h(h), _w(w), _c(c) {
        _img = cv::Mat({ h, w, c }, CV_8UC3);
    };
    Image() {
    };
    ~Image() {};

    void normalize();
    void applyThreshold(double t);
    void zero(); // set all pixels as zero
    void rotate(); // rotate 90 degrees

    void show(char* name);
    void showImageLayers(char* name);

    void makeRandomImage(int h, int w, int c);
    void makeRandomKernel(int size, int c);
    void copyFrom(Image p);
    void loadFromFile(std::string filename);

    double getPixel(int x, int y, int c);
    double getPixelExtend(int x, int y, int c);
    void setPixel(int x, int y, int c, double val);

    Image getImageLayer(int l);

    void twoDConvolve(int mc, Image kernel, int kc, int stride, Image out, int oc);
    void upsample(int stride, Image out);
    void convolve(Image kernel, int stride, int channel, Image out);
    void backConvolve(Image kernel, int stride, int channel, Image out);
    void kernelUpdate(Image update, int stride, int channel, Image out);
private:
    int _h;
    int _w;
    int _c;
    double* _data;
    cv::Mat _img;
};