#include <iostream>
#include "opencv2/opencv.hpp"

int main(void)
{
    cv::Mat img_per_band;
    cv::namedWindow("example");
    img_per_band = cv::imread("test.jpg");  // convert to 8 bit gray scale
    if (img_per_band.empty())
    {
        std::cout << "No image" << std::endl;
        return -1;
    }
    cv::imshow("example", img_per_band);
    cv::waitKey(0);
    cv::destroyWindow("example");
    return 0;
    //cv::Mat img_merge;
    //cv::merge(img_per_band, 5, img_merge);

    //SaliencyGC img(img_merge);

    //img.Histogram();
}