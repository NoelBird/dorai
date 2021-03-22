#pragma once

#include "opencv2/highgui/highgui_c.h"
#include <string>
#include "opencv2/imgproc/imgproc_c.h"

class Image {
public:
    Image(int h, int w, int c) : _h(h), _w(w), _c(c) {
        _data = new double[(long long)h * w * c]; // BUGABLE: overflow 가능성 있음
    };
    Image(): _h(0), _w(0), _c(0), _data(nullptr) {
    };

    Image(Image& p): _h(p._h), _w(p._w), _c(p._c), _data(p._data) {
        // BUGABLE: 복사생성자 이렇게 쓰는게 맞나..?
    };

    
    ~Image() {
        delete[] _data;
    };

    void normalize();
    void applyThreshold(double t);
    void zero(); // set all pixels as zero
    void rotate(); // rotate 90 degrees

    void show(const char* name);
    void showImageLayers(char* name);

    void make(int h, int w, int c);
    void makeRandomImage(int h, int w, int c);
    void makeRandomKernel(int size, int c);
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
};