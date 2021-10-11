#include <iostream>
#include "saliencyGC.h"
#include "opencv2/opencv.hpp"
#include <utility>
#include <cstdlib>

#include <typeinfo>

SaliencyGC::SaliencyGC(cv::Mat &img)
{
    cv::split(img, imgRaw);
    X = img.cols;
    Y = img.rows;
    Channel = img.channels();
    cv::Mat imgNorm[5] = { cv::Mat(X, Y, CV_8U) };
    cv::Mat imgQuant[5] = { cv::Mat(X, Y, CV_8U) };
    imgSal = cv::Mat(X, Y, CV_8U);
    cv::Mat histo[5];
}

int SaliencyGC::GetX()
{
    return X;
}

int SaliencyGC::GetY()
{
    return Y;
}

int SaliencyGC::GetChannel()
{
    return Channel;
}

void SaliencyGC::Normalize()
{
    for (int i = 0; i < Channel; i++)
    {
        cv::normalize(imgRaw[i], imgNorm[i], 255, 0, 32);
    }

    //Initialize check well normalized
    /*
    double minVal;
    double maxVal;

    minMaxLoc(imgNorm[0], &minVal, &maxVal, 0, 0);

    std::cout << minVal << maxVal << std::endl;
    */
}

void SaliencyGC::Histogram(int label)
{
    const int Ch[] = { 0 };
    int histSize = label;
    float channel_range[] = { 0.0, 255.0 };
    const float* channel_ranges = { channel_range };
    cv::calcHist(&imgRaw[0], 1, Ch, cv::Mat(), histo[0], 1, &histSize, &channel_ranges);
    cv::calcHist(&imgRaw[1], 1, Ch, cv::Mat(), histo[1], 1, &histSize, &channel_ranges);
    cv::calcHist(&imgRaw[2], 1, Ch, cv::Mat(), histo[2], 1, &histSize, &channel_ranges);
    cv::calcHist(&imgRaw[3], 1, Ch, cv::Mat(), histo[3], 1, &histSize, &channel_ranges);
    cv::calcHist(&imgRaw[4], 1, Ch, cv::Mat(), histo[4], 1, &histSize, &channel_ranges);

    //return the histogram info
    /*
    std::cout << histo[0].rows << std::endl;
    std::cout << histo[0].cols << std::endl;
    std::cout << histo[0].channels() << std::endl;
    std::cout << histo[0].type() << std::endl;
    std::cout << typeid(histo[0].at<uchar>(0,0)).name() << std::endl;
    */
}

void SaliencyGC::GetSal(int label)
{
    // histogram labels and length
    uchar labels = (uchar)label;
    double length = 255 / labels;

    int total_pixel = X * Y;
    // Saliency value per bands
    cv::Mat Sal_per_band[5] = { cv::Mat::zeros(Y, X, CV_32F), cv::Mat::zeros(Y, X, CV_32F), cv::Mat::zeros(Y, X, CV_32F), cv::Mat::zeros(Y, X, CV_32F), cv::Mat::zeros(Y, X, CV_32F) };

    for (int i = 0; i < Channel; i++){
        for (int y = 0; y < Y; y++) {
            for (int x = 0; x < X; x++) {
                // import pixel value and quantization 
                uchar data = imgNorm[i].at<uchar>(y, x);
                int quant_pixel = (int)data / length;

                // calculate saliency value of pixel
                float sal_value = 0;
                for (int bin = 0; bin < label; bin++) {
                    // frequency of each bin
                    float frequency = ((histo[i].at<float>(bin, 0)) / total_pixel);
                    // sum saliency value (frequency of each bin * distance)
                    sal_value +=  frequency * abs(quant_pixel - bin);
                }
                Sal_per_band[i].at<float>(y, x) = sal_value;
            }
        }
    }
    // add total Saliency value to imgSal_temp. You can modify weight to each band if you need
    cv::Mat imgSal_temp = Sal_per_band[0] + Sal_per_band[1] + Sal_per_band[2] + Sal_per_band[3] + Sal_per_band[4];
    // Normalize value for grayscale image
    cv::normalize(imgSal_temp, imgSal_temp, 255, 0, 32);
    imgSal_temp.convertTo(imgSal, CV_8U);
}

void SaliencyGC::Salshow()
{
    cv::namedWindow("Saliency", CV_WINDOW_AUTOSIZE);
    cv::imshow("Saliency", imgSal);
    cv::waitKey(0);
    cv::destroyWindow("Saliency");
}

/*
void SaliencyGC::Quantitize(int label)
{
    uchar labels = (uchar)label;
    double length = 255 / labels;
    std::cout << length << std::endl;

    for (int i = 0; i < Channel; i++) {
        imgNorm[i].copyTo(imgQuant[i]);
        for (int y = 0; y < Y; y++) {
            for (int x = 0; x < X; x++) {
                uchar data = imgNorm[i].at<uchar>(y,x);
                uchar quant = (uchar) data / length;
                imgQuant[i].at<uchar>(y, x) = quant;
            }
        }
    }
}
*/