#include <iostream>
#include "saliencyGC.h"
#include "opencv2/opencv.hpp"
#include <ctime>

int main(void)
{
    const int num_bins = 4;
    const int band = 5;

    const char* path = "000/IMG_0095_*.tif";
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
    // Check image size
    // std::cout << img.GetX() << std::endl;
    // std::cout << img.GetY() << std::endl;
    // std::cout << img.GetChannel() << std::endl;
    
    img.Resize(0.5); // Consider Original Size (960*1280)

    img.Normalize();

    img.Histogram(num_bins);

    img.GetSal(num_bins);

    // end time calculation
    std::cout << float(clock() - begin_time) / CLOCKS_PER_SEC;

    img.Salshow();
}
