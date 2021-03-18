#include "image.h"

#include <opencv2/opencv.hpp>

void Image::loadFromFile(std::string filename)
{
    cv::Mat src = cv::Mat();
    src = cv::imread(filename);
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
    _img = cv::Mat(h, w, c);
    int i, j, k, count = 0;;

    for (k = 0; k < c; ++k) {
        for (i = 0; i < h; ++i) {
            for (j = 0; j < w; ++j) {
                _img.data[count++] = data[i * step + j * c + k];
            }
        }
    }
    src.release();
    return;
}