#include "image.h"

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <cassert>

void Image::make(int h, int w, int c)
{
    _h = h;
    _w = w;
    _c = c;

    _data = new double[_h * _w * _c];
}

Image* Image::makeRandomImage(int h, int w, int c)
{
    Image* out = new Image(h, w, c);
    int i;
    for (i = 0; i < h * w * c; ++i) {
        out->_data[i] = (double)rand() / RAND_MAX;
    }
    return out;
}

void Image::loadFromFile(std::string filename)
{
    cv::Mat src = cv::imread(filename);
    if(src.empty())
    {
        printf("Cannot load file image: %s\n", filename.c_str());
        exit(0);
    }

    unsigned char* data = src.data;
    int c = src.channels();
    int h = src.size().height;
    int w = src.size().width;
    int step = src.step;
    int i, j, k, count = 0;

    this->make(h, w, c);
    this->zero();

    for (k = 0; k < c; ++k) {
        for (i = 0; i < h; ++i) {
            for (j = 0; j < w; ++j) {
                _data[count++] = data[i * step + j * c + k];
            }
        }
    }
    src.release();
    return;
}

double Image::getPixel(int x, int y, int c)
{
    assert(x < _h&& y < _w&& c < _c);
    return _data[c * _h * _w + x * _w + y];
}

void Image::setPixel(int x, int y, int c, double val)
{
    assert(x < _h&& y < _w&& c < _c);
    _data[c * _h * _w + x * _w + y] = val;
}

void Image::normalize()
{
    double* min = new double[_c];
    double* max = new double[_c];
    int i, j;
    for (i = 0; i < _c; ++i) min[i] = max[i] = _data[i * _h * _w];

    for (j = 0; j < _c; ++j) {
        for (i = 0; i < _h * _w; ++i) {
            double v = _data[i + j * _h * _w];
            if (v < min[j]) min[j] = v;
            if (v > max[j]) max[j] = v;
        }
    }
    for (i = 0; i < _c; ++i) {
        if (max[i] - min[i] < .00001) {
            min[i] = 0;
            max[i] = 1;
        }
    }
    for (j = 0; j < _c; ++j) {
        for (i = 0; i < _w * _h; ++i) {
            _data[i + j * _h * _w] = (_data[i + j * _h * _w] - min[j]) / (max[j] - min[j]);
        }
    }

    delete[] min;
    delete[] max;
}

void Image::zero()
{
    memset(_data, 0, _h * _w * _c * sizeof(double)); // overflow 가능성 있음
}

void Image::rotate()
{
    int i, j;
    for (j = 0; j < _c; ++j) {
        for (i = 0; i < _h * _w / 2; ++i) {
            double swap = _data[j * _h * _w + i];
            _data[j * _h * _w + i] = _data[j * _h * _w + (_h * _w - 1 - i)];
            _data[j * _h * _w + (_h * _w - 1 - i)] = swap;
        }
    }
}

void Image::show(const char* name)
{

    int windows = 0;
    int i, j, k;
    Image copy(*this);
    copy.normalize();

    char buff[256];
    sprintf_s(buff, "%s (%d)", name, windows);

    cv::Mat disp = cv::Mat(cvSize(_w, _h), CV_8UC(_c));
    int step = disp.step;
    cvNamedWindow(buff, CV_WINDOW_AUTOSIZE);
    cvMoveWindow(buff, 100 * (windows % 10) + 200 * (windows / 10), 100 * (windows % 10));
    ++windows;
    for (i = 0; i < _h; ++i) {
        for (j = 0; j < _w; ++j) {
            for (k = 0; k < _c; ++k) {
                disp.data[i * step + j * _c + k] = (unsigned char)(copy.getPixel(i, j, k) * 255);
            }
        }
    }
    if (disp.size().height < 100 || disp.size().width < 100) {
        cv::Mat* buffer = &disp;
        cv::resize(*buffer, disp, cvSize(100, 100 * _h / _w), 0, 0, cv::INTER_NEAREST);
    }
    cv::imshow(buff, disp);
}

void Image::showImageLayers(const char* name)
{
    int i;
    char buff[256];
    for (i = 0; i < _c; ++i) {
        sprintf_s(buff, "%s - Layer %d", name, i);
        Image* layer = getImageLayer(i);
        layer->show(buff);
    }
}

void Image::upsample(int stride, Image* out)
{

    int i, j, k;
    out->zero();
    for (k = 0; k < _c; ++k) {
        for (i = 0; i < _h; ++i) {
            for (j = 0; j < _w; ++j) {
                double val = this->getPixel(i, j, k);
                out->setPixel(i * stride, j * stride, k, val);
            }
        }
    }
}
