#include <iostream>
#include "opencv2/opencv.hpp"
#include <ctime>
#include <utility>
#include <cstdlib>
#include <typeinfo>

#define k_means 4

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
    cv::Mat histo[k_means][5];
    cv::Mat segment_labels;
    cv::Mat Salcut;
public:
    SaliencyGC(cv::Mat img[5]);
    int GetX();
    int GetY();
    int GetChannel();
    void Resize(double resize_factor);
    void Histogram(int label);
    void Normalize();
    void Segment();
    void GetSal(int label);
    void Salshow();
};

int main(void)
{
    const int num_bins = 6;
    const int band = 5;

    const char* path = "data/000/IMG_0075_*.tif";
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

    SaliencyGC img(img_per_band);    

    img.Resize(0.5); // Consider Original Size (960*1280)

    img.Normalize();

    img.Segment();

    img.Histogram(num_bins);

    img.GetSal(num_bins);

    // end time calculation
    std::cout << float(clock() - begin_time) / CLOCKS_PER_SEC << std::endl;

    img.Salshow();

}


SaliencyGC::SaliencyGC(cv::Mat img[5])
{
    for (int i=0; i<5; i++)
    {
        imgRaw[i] = img[i];
    }
    X = imgRaw[0].cols;
    Y = imgRaw[0].rows;
    Channel = 5;
    cv::Mat imgNorm[5] = { cv::Mat(X, Y, CV_8U) };
    cv::Mat imgQuant[5] = { cv::Mat(X, Y, CV_8U) };
    imgSal = cv::Mat(X, Y, CV_8U);
    cv::Mat histo[k_means][5];
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
}

void SaliencyGC::Histogram(int label)
{
    const int Ch[] = { 0 };
    int histSize = label;
    float channel_range[] = { 0.0, 255.0 };
    const float* channel_ranges = { channel_range };
    for(int i=0;i<k_means;i++)
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
            // cv::Mat masked_img = cv::Mat::zeros(Y,X,CV_32F);
            // imgRaw[j].copyTo(masked_img,mask);
            cv::calcHist(&imgRaw[j], 1, Ch, mask, histo[i][j], 1, &histSize, &channel_ranges);
        }
        
    }
    
}

void SaliencyGC::GetSal(int num_bins)
{
    
    float sal[k_means] = {0.0};
    int total_pixel[k_means][Channel] = {0};

    for (int i=0; i<k_means; i++){
        for (int c=0; c<Channel; c++){
            for (int m=0 ;m<num_bins; m++){
                total_pixel[i][c] += (int) histo[i][c].at<float>(m,0);
            }
            
        }
    }
    for (int i=0;i<k_means;i++){
        for (int c=0;c<Channel;c++){
            for (int j=0;j<k_means;j++){
                if (i != j){
                    for (int m = 0; m<num_bins; m++){
                        for (int n = 0; n<num_bins; n++){
                            float freq1 = histo[i][c].at<float>(m,0)/total_pixel[i][c];
                            float freq2 = histo[j][c].at<float>(n,0)/total_pixel[j][c];
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
    // imgSal = 255-imgSal;

    cv::threshold(imgSal, Salcut, 250, 255, 0);

}

void SaliencyGC::Salshow()
{

    cv::namedWindow("Saliency", CV_WINDOW_AUTOSIZE);
    cv::imshow("Saliency", imgSal);
    cv::waitKey(0);
    cv::destroyWindow("Saliency");

    // cv::imwrite("Saliency map.png",imgSal);
    // cv::imwrite("Saleincy cut.png", Salcut);
}

void SaliencyGC::Segment()
{
    cv::Mat flatten = cv::Mat::zeros(X * Y, Channel+2, CV_32F);
    
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
            flatten.at<float>(x + y * X, 5) = x / X;
            flatten.at<float>(x + y * X, 6) = y / Y;
        }
    }

    cv::Mat centeres, New_image;
    int attempts = 5;
    cv::kmeans(flatten, k_means, segment_labels, cv::TermCriteria(CV_TERMCRIT_ITER + CV_TERMCRIT_EPS, 10, 1.0), attempts, cv::KMEANS_PP_CENTERS, centeres);

    
    int colors[k_means];
    for (int i=0;i<k_means;i++)
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

    /*
    cv::namedWindow("K-means", CV_WINDOW_AUTOSIZE);
    cv::imshow("K-means", New_image);
    cv::waitKey(0);
    cv::destroyWindow("K-means");

    cv::imwrite("K-means clustering.png", New_image);
    */
}
