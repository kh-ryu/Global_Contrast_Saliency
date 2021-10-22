#include <iostream>
#include "opencv2/opencv.hpp"
#include <ctime>
#include <utility>
#include <cstdlib>
#include <typeinfo>

#define k_means 6

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
    cv::Mat histo[5][k_means];
    cv::Mat segment_labels;
public:
    SaliencyGC(cv::Mat& img);
    int GetX();
    int GetY();
    int GetChannel();
    void Resize(double resize_factor);
    void Histogram(int label, int k);
    void Normalize();
    void Segment(int k);
    void GetSal(int label);
    void Salshow();
};

int main(void)
{
    const int num_bins = 4;
    const int band = 5;

    const char* path = "Opencv_test/000/IMG_0123_*.tif";
    std::vector<cv::String> filenames;
    cv::glob(path, filenames, false);
    cv::Mat img_per_band[band];

    for (int i = 0; i < filenames.size(); i++)
    {
        cv::String filename = filenames[i];
        img_per_band[i] = cv::imread(filename, 0);  // convert to 8 bit gray scale
    }

    if (img_per_band[0].empty())
    {
        std::cout << "No image" << std::endl;
        return -1;
    }
    // Check loaded image
    cv::namedWindow("Blue", CV_WINDOW_AUTOSIZE);
    cv::imshow("Blue", img_per_band[0]);
    cv::waitKey(0);
    cv::destroyWindow("example");

    // Start time calculation
    const clock_t begin_time = clock();

    // merge images to one data
    cv::Mat img_merge;
    cv::merge(img_per_band, band, img_merge);
    SaliencyGC img(img_merge);
    

    img.Resize(0.5); // Consider Original Size (960*1280)

    img.Normalize();

    img.Segment(k_means);

    img.Histogram(num_bins, k_means);

    img.GetSal(num_bins);

    // end time calculation
    std::cout << float(clock() - begin_time) / CLOCKS_PER_SEC << std::endl;

    img.Salshow();

}


SaliencyGC::SaliencyGC(cv::Mat& img)
{
    cv::split(img, imgRaw);
    X = img.cols;
    Y = img.rows;
    Channel = img.channels();
    cv::Mat imgNorm[5] = { cv::Mat(X, Y, CV_8U) };
    cv::Mat imgQuant[5] = { cv::Mat(X, Y, CV_8U) };
    imgSal = cv::Mat(X, Y, CV_8U);
    cv::Mat histo[5];
    cv::Mat segment_labels;
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

void SaliencyGC::Resize(double resize_factor)
{
    X = X * resize_factor;
    Y = Y * resize_factor;
    for (int i = 0; i < Channel; i++)
    {
        cv::resize(imgRaw[i], imgRaw[i], cv::Size(X, Y));
    }
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

void SaliencyGC::Histogram(int label, int k)
{
    const int Ch[] = { 0 };
    int histSize = label;
    float channel_range[] = { 0.0, 255.0 };
    const float* channel_ranges = { channel_range };
    for(int i=0;i<k;i++)
    {
        cv::Mat mask = cv::Mat::zeros(Y,X,CV_8U);
        for (int y = 0; y < Y; y++) {
            for (int x = 0; x < X; x++) {
                if(segment_labels.at<int>(0,x + y*X)==i){
                    mask.at<uchar>(y,x) = 255;
                }
            }
        }
        
        for(int j=0;j<Channel;j++){
            cv::Mat masked_img = cv::Mat::zeros(Y,X,CV_32F);
            imgRaw[j].copyTo(masked_img,mask);
            cv::calcHist(&masked_img, 1, Ch, cv::Mat(), histo[j][i], 1, &histSize, &channel_ranges);
        }
        
    }
    
}

void SaliencyGC::GetSal(int num_bins)
{
    
    float sal[k_means] = {0.0};

    for (int i=0;i<k_means;i++){
        for (int c=0;c<Channel;c++){
            for (int j=0;j<k_means;j++){
                if (i != j){
                    for (int m = 0; m<num_bins; m++){
                        for (int n = 0; n<num_bins; n++){
                            float freq1 = histo[c][i].at<float>(m,0)/(X*Y);
                            float freq2 = histo[c][j].at<float>(n,0)/(X*Y);
                            sal[i] += freq1*freq2*abs(m-n);
                        }
                    }
                }
            }
        }
    }

    cv::Mat imgSal_temp = cv::Mat(Y, X, CV_32F);
    for (int y = 0; y < Y; y++) {
        for (int x = 0; x < X; x++) {
            imgSal_temp.at<float>(y,x) = (float)(sal[segment_labels.at<int>(0,x + y*X)]);
        }
    }
    
    // Normalize value for grayscale image
    cv::normalize(imgSal_temp, imgSal_temp, 255, 0, 32);
    imgSal_temp.convertTo(imgSal, CV_8U);
    imgSal = 255-imgSal;
}

void SaliencyGC::Salshow()
{
    cv::namedWindow("Saliency", CV_WINDOW_AUTOSIZE);
    cv::imshow("Saliency", imgSal);
    cv::waitKey(0);
    cv::destroyWindow("Saliency");

    cv::imwrite("0123 Saleincy.png",imgSal);
}

void SaliencyGC::Segment(int k)
{
    cv::Mat flatten = cv::Mat::zeros(X * Y, Channel, CV_32F);
    
    cv::blur(imgNorm[0],imgNorm[0],cv::Size(10,10));
    cv::blur(imgNorm[1],imgNorm[1],cv::Size(10,10));
    cv::blur(imgNorm[2],imgNorm[2],cv::Size(10,10));
    cv::blur(imgNorm[3],imgNorm[3],cv::Size(10,10));
    cv::blur(imgNorm[4],imgNorm[4],cv::Size(10,10));
    
    for (int y = 0; y < Y; y++) {
        for (int x = 0; x < X; x++) {
            flatten.at<float>(x + y * X, 0) = imgNorm[0].at<uchar>(y, x) / 255.0;
            flatten.at<float>(x + y * X, 1) = imgNorm[1].at<uchar>(y, x) / 255.0;
            flatten.at<float>(x + y * X, 2) = imgNorm[2].at<uchar>(y, x) / 255.0;
            flatten.at<float>(x + y * X, 3) = imgNorm[3].at<uchar>(y, x) / 255.0;
            flatten.at<float>(x + y * X, 4) = imgNorm[4].at<uchar>(y, x) / 255.0;
        }
    }

    cv::Mat centeres, New_image;
    int attempts = 10;
    cv::kmeans(flatten, k, segment_labels, cv::TermCriteria(CV_TERMCRIT_ITER + CV_TERMCRIT_EPS, 10, 1.0), attempts, cv::KMEANS_PP_CENTERS, centeres);

    int colors[k];
    for (int i=0;i<k;i++)
    {
        colors[i] = 255/(i+1);
    }

    New_image = cv::Mat(Y, X, CV_32F);
    for (int y = 0; y < Y; y++) {
        for (int x = 0; x < X; x++) {
            New_image.at<float>(y,x) = (float)(colors[segment_labels.at<int>(0,x + y*X)]);
        }
    }

    New_image.convertTo(New_image, CV_8U);

    cv::namedWindow("K-means", CV_WINDOW_AUTOSIZE);
    cv::imshow("K-means", New_image);
    cv::waitKey(0);
    cv::destroyWindow("K-means");
}