#include <iostream>
#include "opencv2/opencv.hpp"
#include <ctime>
#include <utility>
#include <cstdlib>
#include <typeinfo>

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
    cv::Mat Salcut;
public:
    SaliencyGC(cv::Mat img[5]);
    int GetX();
    int GetY();
    int GetChannel();
    void Resize(double resize_factor);
    void Histogram(int label);
    void Normalize();
    void GetSal(int label);
    void Salshow();
};

int main(void)
{
    // number of bins for color histogram
    const int num_bins = 8;
    // number of bands in Micasense image
    const int band = 5;

    const char* path = "data/000/IMG_0075_*.tif";
    std::vector<cv::String> filenames;
    cv::glob(path, filenames, false);
    cv::Mat img_per_band[band];

    for(int i=0; i<filenames.size(); i++)
    {
        cv::String filename = filenames[i];
        img_per_band[i] = cv::imread(filename, 0);  // convert to 8 bit gray scale
    }
    
    if (img_per_band[0].empty())
    {
        std::cout << "No image" << std::endl;
        return -1;
    }
    
    // Start time calculation
    const clock_t begin_time = clock();

    // merge images to one data
    // cv::Mat img_merge;
    // cv::merge(img_per_band, band, img_merge);
    SaliencyGC img(img_per_band);

    img.Resize(0.5); // Consider Original Size (960*1280)

    img.Normalize();

    img.Histogram(num_bins);

    img.GetSal(num_bins);

    // end time calculation
    std::cout << float(clock() - begin_time) / CLOCKS_PER_SEC << std::endl;

    img.Salshow();

    return 0;
}

// initialize
SaliencyGC::SaliencyGC(cv::Mat img[5])
{
    for (int i=0; i<5; i++)
    {
        imgRaw[i] = img[i];
    }
    X = imgRaw[0].cols;
    Y = imgRaw[0].rows;
    Channel = imgRaw[0].channels();
    cv::Mat imgNorm[5] = { cv::Mat(X, Y, CV_8U) };
    cv::Mat imgQuant[5] = { cv::Mat(X, Y, CV_8U) };
    imgSal = cv::Mat(X, Y, CV_8U);
    cv::Mat histo[5];
}

// return X of image size (column)
int SaliencyGC::GetX()
{
    return X;
}

// return Y of image size (rows)
int SaliencyGC::GetY()
{
    return Y;
}

// return the number of channels
int SaliencyGC::GetChannel()
{
    return Channel;
}

// resize image size
void SaliencyGC::Resize(double resize_factor)
{
    X = X * resize_factor;
    Y = Y * resize_factor;
    for (int i = 0; i < Channel; i++)
    {
        cv::resize(imgRaw[i], imgRaw[i], cv::Size(X,Y));
    }
}

// normalize image to 8 bit gray scale
void SaliencyGC::Normalize()
{
    for (int i = 0; i < Channel; i++)
    {
        cv::normalize(imgRaw[i], imgNorm[i], 255, 0, 32);
    }
}

// calculate color histogram of each band
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
}

// calculate saliency value of each pixel
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
    cv::normalize(imgSal_temp, imgSal_temp, 0, 255, 32);
    imgSal_temp.convertTo(imgSal, CV_8U);
    // if saliency value is above 170, it is saliency object
    cv::threshold(imgSal,Salcut,200,255,0);
}

// show saliency map
void SaliencyGC::Salshow()
{
    cv::namedWindow("Saliency", CV_WINDOW_AUTOSIZE);
    cv::imshow("Saliency", Salcut);
    cv::waitKey(0);
    cv::destroyWindow("Saliency");
}


