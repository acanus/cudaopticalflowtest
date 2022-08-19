// For tutorial go to
// https://funvision.blogspot.com
#include <iostream>
#include <fstream>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/core.hpp>
#include <opencv2/video.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/cudabgsegm.hpp>
#include <opencv2/cudaobjdetect.hpp>
#include <opencv2/cudaoptflow.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include "opencv2/cudaarithm.hpp"
#include <omp.h>
#include <chrono>

using namespace std;
using namespace cv;
using namespace cv::cuda;
template <typename T>
static cv::Ptr<T>* clone(const cv::Ptr<T>& ptr)
{
    return new cv::Ptr<T>(ptr);
}

extern "C" __declspec(dllexport) void CreateFarnebackOpticalFlow(Ptr<cuda::FarnebackOpticalFlow> **returnValue, 
                                                                                                int numLevels,
                                                                                                double pyrScale ,
                                                                                                bool fastPyramids ,
                                                                                                int winSize ,
                                                                                                int numIters ,
                                                                                                int polyN ,
                                                                                                double polySigma ,
                                                                                                int flags )
{
    *returnValue = clone(cuda::FarnebackOpticalFlow::create(numLevels ,
                                                             pyrScale ,
                                                             fastPyramids ,
                                                             winSize ,
                                                             numIters,
                                                             polyN ,
                                                             polySigma,
                                                             flags));
}
extern "C" __declspec(dllexport) void FarnebackOpticalFlowGet(cv::Ptr<cv::cuda::FarnebackOpticalFlow> *ptr, cv::cuda::FarnebackOpticalFlow **returnValue)
{
    *returnValue = ptr->get();
}
extern "C" __declspec(dllexport) void CreateOpticalFlowDual_TVL1(Ptr<cuda::OpticalFlowDual_TVL1> **returnValue, 
    double tau = 0.25,
    double lambda = 0.15,
    double theta = 0.3,
    int nscales = 5,
    int warps = 5,
    double epsilon = 0.01,
    int iterations = 300,
    double scaleStep = 0.8,
    double gamma = 0.0,
    bool useInitialFlow = false)
{
    *returnValue = clone(cuda::OpticalFlowDual_TVL1::create( 
         tau ,
         lambda ,
         theta ,
         nscales,
         warps ,
         epsilon ,
         iterations,
         scaleStep ,
         gamma,
         useInitialFlow ));
}
extern "C" __declspec(dllexport) void OpticalFlowDual_TVL1Get(cv::Ptr<cv::cuda::OpticalFlowDual_TVL1> *ptr, cv::cuda::OpticalFlowDual_TVL1 * *returnValue)
{
    *returnValue = ptr->get();
}
extern "C" __declspec(dllexport) void CreateOpticalFlowBrox(Ptr<cuda::BroxOpticalFlow> **returnValue,
    double alpha = 0.197,
    double gamma = 50.0,
    double scale_factor = 0.8,
    int inner_iterations = 5,
    int outer_iterations = 150,
    int solver_iterations = 10)
{
    *returnValue = clone(cuda::BroxOpticalFlow::create(
        alpha,
        gamma ,
        scale_factor ,
        inner_iterations,
        outer_iterations ,
        solver_iterations ));
}
extern "C" __declspec(dllexport) void OpticalFlowBroxGet(cv::Ptr<cv::cuda::BroxOpticalFlow> *ptr, cv::cuda::BroxOpticalFlow * *returnValue)
{
    *returnValue = ptr->get();
}
extern "C" __declspec(dllexport) void DenseOpticalFlowCalc(cuda::DenseOpticalFlow *obj, uint8_t* image1, uint8_t * image2,Mat **map_vector,uint32_t w, uint32_t h)
{
    Mat flow;
    Mat mat1(h, w, CV_8UC1, image1);
    Mat mat2(h, w, CV_8UC1, image2);
    GpuMat gflow(mat1.size(), CV_32FC2);
    GpuMat GpuImg0(mat1);
    GpuMat GpuImg1(mat2);
    obj->calc(GpuImg0, GpuImg1, gflow);
    gflow.download(flow);
    Mat map(flow.size(), CV_32FC2);
    for (int y = 0; y < map.rows; ++y)
    {
        for (int x = 0; x < map.cols; ++x)
        {
            Point2f f = flow.at<Point2f>(y, x);
            map.at<Point2f>(y, x) = Point2f(x + f.x, y + f.y);
        }
    }
    *map_vector = new cv::Mat(map);
}
extern "C" __declspec(dllexport) void DenseOpticalFlowCalcFloat(cuda::DenseOpticalFlow * obj, uint8_t * image1, uint8_t * image2, Mat * *map_vector, uint32_t w, uint32_t h)
{
    Mat flow;
    Mat mat1(h, w, CV_8UC1, image1);
    Mat mat2(h, w, CV_8UC1, image2);
    Mat mat1float;
    Mat mat2float;
    mat1.convertTo(mat1float, CV_32FC1, 1 / 255.0);
    mat2.convertTo(mat2float, CV_32FC1, 1 / 255.0);
    GpuMat gflow(mat1float.size(), CV_32FC2);
    GpuMat GpuImg0(mat1float);
    GpuMat GpuImg1(mat2float);
    obj->calc(GpuImg0, GpuImg1, gflow);
    gflow.download(flow);
    Mat map(flow.size(), CV_32FC2);
    for (int y = 0; y < map.rows; ++y)
    {
        for (int x = 0; x < map.cols; ++x)
        {
            Point2f f = flow.at<Point2f>(y, x);
            map.at<Point2f>(y, x) = Point2f(x + f.x, y + f.y);
        }
    }
    *map_vector = new cv::Mat(map);
}
extern "C" __declspec(dllexport) void Remap(Mat * map, uint8_t * image, uint8_t * imageMapped, int32_t w, int32_t h)
{
    Mat imagemat(h, w, CV_8UC1, image);
    Mat imageMappedMat(h, w, CV_8UC1, imageMapped);
    remap(imagemat, imageMappedMat, *map, cv::Mat(), INTER_AREA);
}

//void FarnebackOpticalFlowCalc(cuda::FarnebackOpticalFlow* obj, uint8_t* image1, uint8_t* image2, uint8_t* UnwrappedImage, int32_t w, int32_t h)
//{
//    Mat flow;
//
//    Mat mat1(w, h, CV_8UC1, image1);
//    Mat mat2(w, h, CV_8UC1, image2);
//    Mat UnwrappedMat(w, h, CV_8UC1, UnwrappedImage);
//    GpuMat gflow(mat1.size(), CV_32FC2);
//    GpuMat GpuImg0(mat1);
//    GpuMat GpuImg1(mat2);
//    obj->calc(GpuImg0, GpuImg1, gflow);
//    gflow.download(flow);
//    Mat map(flow.size(), CV_32FC2);
//    for (int y = 0; y < map.rows; ++y)
//    {
//        for (int x = 0; x < map.cols; ++x)
//        {
//            Point2f f = flow.at<Point2f>(y, x);
//            map.at<Point2f>(y, x) = Point2f(x + f.x, y + f.y);
//        }
//    }
//    remap(mat1, UnwrappedMat, map, NULL, INTER_AREA);
//}
int main()
{

    
    //Mat frame1;
    Mat frame0 = cv::imread("C:\\src\\cudaopticalflow\\cudaopticalflowtest\\images\\1.png", cv::ImreadModes::IMREAD_GRAYSCALE);
    Mat frame1 = cv::imread("C:\\src\\cudaopticalflow\\cudaopticalflowtest\\images\\2.png", cv::ImreadModes::IMREAD_GRAYSCALE);
    Mat frame1Float;
    frame0.convertTo(frame0, CV_32FC1, 1 / 255.0);
    frame1.convertTo(frame1Float, CV_32FC1, 1 / 255.0);
    // https://funvision.blogspot.com
    //Ptr<cuda::FarnebackOpticalFlow> farn = cuda::FarnebackOpticalFlow::create(10,0.5,false,100,10,7,1.5);
    Ptr<cuda::BroxOpticalFlow> farn = cuda::BroxOpticalFlow::create(20,50,0.5,1,150,10);
    
    Mat flow;
    //Put Mat into GpuMat
    GpuMat GpuImg0(frame0);
    GpuMat GpuImg1(frame1Float);
    GpuMat gflow(frame0.size(), CV_32FC2);
    farn->calc(GpuImg0, GpuImg1, gflow);
    gflow.download(flow);
    Mat map(flow.size(), CV_32FC2);
    for (int y = 0; y < map.rows; ++y)
    {
        for (int x = 0; x < map.cols; ++x)
        {
            Point2f f = flow.at<Point2f>(y, x);
            map.at<Point2f>(y, x) = Point2f(x + f.x, y + f.y);
        }
    }
    Mat imageMappedMat(frame0.size(), CV_8UC1);
    remap(frame1, imageMappedMat, map, cv::Mat(), INTER_AREA);
    cv::imwrite("C:\\src\\cudaopticalflow\\cudaopticalflowtest\\images\\3.png",imageMappedMat);
    
}
int main2()
{

    Mat frame0;
    Mat frame1;
    // https://funvision.blogspot.com
    Ptr<cuda::OpticalFlowDual_TVL1> farn = cuda::OpticalFlowDual_TVL1::create(0.25,0.15,0.3,10,10,0.1,100);
    /*uint8_t byte1[100 * 100];
    uint8_t byte2[100 * 100];
    uint8_t byte3[100 * 100];*/
    //FarnebackOpticalFlowCalc(farn, byte1, byte2, byte3, 100, 100);
    //Capture camera
    VideoCapture cap(0);

    for (;;) {
        Mat image;
        cap >> image;

        cv::cvtColor(image, frame0, cv::COLOR_BGR2GRAY);
        //cv::resize(frame0, frame0, cv::Size(0,0), 3, 3, 1);
        if (frame1.empty()) {
            frame0.copyTo(frame1);
        }
        else {
            Mat flow;
            //Put Mat into GpuMat
            GpuMat GpuImg0(frame0);
            GpuMat GpuImg1(frame1);
            //Prepare space for output
            GpuMat gflow(frame0.size(), CV_32FC2);
            // chrono time to calculate the the needed time to compute and
            // draw the optical flow result
            std::chrono::steady_clock::time_point begin =
                std::chrono::steady_clock::now();
            // Calculate optical flow
            farn->calc(GpuImg0, GpuImg1, gflow);
            // GpuMat to Mat
            gflow.download(flow);

            for (int y = 0; y < image.rows - 1; y += 10) {
                for (int x = 0; x < image.cols - 1; x += 10) {
                    // get the flow from y, x position * 10 for better visibility
                    const Point2f flowatxy = flow.at<Point2f>(y, x) * 5;
                    // draw line at flow direction
                    line(image, Point(x, y), Point(cvRound(x + flowatxy.x),
                        cvRound(y + flowatxy.y)), Scalar(0, 255, 0), 2);
                    // draw initial point  https://funvision.blogspot.com
                    circle(image, Point(x, y), 1, Scalar(0, 0, 255), -1);
                }
            }

            // end - begin time to calculate compute farneback opt flow + draw
            std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
            std::cout << "Time difference = " <<
                std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count()
                << "[ms]" << std::endl;
            // Display result  https://funvision.blogspot.com
            imshow("Display window", image);
            waitKey(25);
            // Save frame0 to frame1 to for next round
            // https://funvision.blogspot.com
            frame0.copyTo(frame1);
        }
    }
}