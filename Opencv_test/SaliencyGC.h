#pragma once

#ifndef __SALIENCYGC_H__
#define __SALIENCYGC_H__

#include "opencv2/opencv.hpp"

class SaliencyGC
{
private:
    cv::Mat imgRaw[5];
    cv::Mat imgNorm[5];
    cv::Mat imgQuant[5];
    cv::Mat imgSal;
    int X;
    int Y;
    int Channel;
    cv::Mat histo[5];
public:
    SaliencyGC(cv::Mat &img);
    int GetX();
    int GetY();
    int GetChannel();
    void Histogram(int label);
    void Normalize();
    // void Quantitize(int label);
    void GetSal(int label);
    void Salshow();
};

#endif