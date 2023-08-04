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
#include <opencv2/mcc.hpp>
#include "opencv2/mcc/ccm.hpp"

using namespace cv::mcc;
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
//int main3()
//{
//
//    
//    //Mat frame1;
//    Mat frame0 = cv::imread("C:\\src\\cudaopticalflow\\cudaopticalflowtest\\images\\1.png", cv::ImreadModes::IMREAD_GRAYSCALE);
//    Mat frame1 = cv::imread("C:\\src\\cudaopticalflow\\cudaopticalflowtest\\images\\2.png", cv::ImreadModes::IMREAD_GRAYSCALE);
//    Mat frame1Float;
//    frame0.convertTo(frame0, CV_32FC1, 1 / 255.0);
//    frame1.convertTo(frame1Float, CV_32FC1, 1 / 255.0);
//    // https://funvision.blogspot.com
//    //Ptr<cuda::FarnebackOpticalFlow> farn = cuda::FarnebackOpticalFlow::create(10,0.5,false,100,10,7,1.5);
//    Ptr<cuda::BroxOpticalFlow> farn = cuda::BroxOpticalFlow::create(20,50,0.5,1,150,10);
//    
//    Mat flow;
//    //Put Mat into GpuMat
//    GpuMat GpuImg0(frame0);
//    GpuMat GpuImg1(frame1Float);
//    GpuMat gflow(frame0.size(), CV_32FC2);
//    farn->calc(GpuImg0, GpuImg1, gflow);
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
//    Mat imageMappedMat(frame0.size(), CV_8UC1);
//    remap(frame1, imageMappedMat, map, cv::Mat(), INTER_AREA);
//    cv::imwrite("C:\\src\\cudaopticalflow\\cudaopticalflowtest\\images\\3.png",imageMappedMat);
//    
//}
static Mat getColorCheckerMASK(const uchar* checker, int row)
{
    Mat res(row, 1, CV_8U);
    for (int i = 0; i < row; ++i)
    {
        res.at<uchar>(i, 0) = checker[i];
    }
    return res;
}

static Mat getColorChecker(const double* checker, int row)
{
    Mat res(row, 1, CV_64FC3);
    for (int i = 0; i < row; ++i)
    {
        res.at<Vec3d>(i, 0) = Vec3d(checker[3 * i], checker[3 * i + 1], checker[3 * i + 2]);
    }
    return res;
}
static const double ColorChecker2005_LAB_D50_2[24][3] = { 
        { 115, 82, 68},
        { 194, 150, 130 },
        { 98, 122, 157 },
        {87, 108, 67 },
        { 133, 128,177},
        { 103, 189, 170 },
        { 214, 126, 44 },
        { 80, 91, 166 },
        { 193, 90, 99 },
        { 94 ,60, 108  },
        { 157, 188 ,64 },
        { 224 ,163, 46},
        { 56 ,61,150 },
        { 70, 148, 73 },
        { 175 ,54, 60 },
        { 231 ,199 ,31 },
        { 187 ,86, 149 },
        { 8 ,133 ,161 },
        { 243 ,243 ,242 },
        { 200 ,200, 200 },
        { 160 ,160 ,160 },
        { 122 ,122, 121 },
        { 85, 85 ,85 },
        { 52, 52 ,52} };
static const uchar ColorChecker2005_COLORED_MASK[24] = { 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1,
        0, 0, 0, 0, 0, 0 };

static Mat getCheckColor() {
    Mat a = Mat(24, 3, CV_64FC1);
    std::memcpy(a.data, ColorChecker2005_LAB_D50_2, 24 * 3 * sizeof(double));
    return a;
}
extern "C" __declspec(dllexport) int CalcColorCorectionMatrix(uint8_t * image, int32_t w, int32_t h, uint8_t * imageDraw, double_t * *ccm)
{
    TYPECHART chartType = TYPECHART(0);
    Mat imagemat(h, w, CV_8UC3, image);
    Mat imagedraw(h, w, CV_8UC3, imageDraw);
    Ptr<CCheckerDetector> detector = CCheckerDetector::create();
    if (!detector->process(imagemat, chartType, 1))
    {
        return 0;
    }
    std::vector<Ptr<mcc::CChecker>> checkers = detector->getListColorChecker();
    if (checkers.size() > 0) {
        auto checker = checkers.front();
        Ptr<CCheckerDraw> cdraw = CCheckerDraw::create(checker);
        cdraw->draw(imagedraw);
        Mat chartsRGB = checker->getChartsRGB();
        Mat src = chartsRGB.col(1).clone().reshape(1, chartsRGB.rows / 3);
        src /= 255.0;
        Mat trgFloat = getCheckColor() / 255.0;
        cv::Mat ccm1;
        cv::solve(src, trgFloat, ccm1, DECOMP_SVD);
        *ccm = ccm1.ptr<double_t>(0);
        return 1;
    }
    return 0;
}
extern "C" __declspec(dllexport) int ApplyCCM(uint8_t * image, int32_t w, int32_t h, double_t * ccm , uint8_t  * image_out)
{
    Mat img_;
    Mat imagemat(h, w, CV_8UC3, image);
    Mat imageoutMat(h, w, CV_8UC3, image_out);
    Mat ccm1(3, 3, CV_64F, ccm);
    cv::cvtColor(imagemat, img_, COLOR_BGR2RGB);
    //cv::cvtColor(image, cieImage, COLOR_BGR2Lab);
    img_.convertTo(img_, CV_64F);
    const int inp_size = 255;
    const int out_size = 255;
    img_ = img_ / inp_size;
    cv::Mat cam1Reshaped = img_.reshape(1, img_.size().height * img_.size().width);
    //Mat calibratedImage = model1.infer(img_);
    Mat calibratedImage = cam1Reshaped * ccm1.reshape(0, 3);
    calibratedImage = calibratedImage.reshape(3, img_.size().height);
    Mat out_ = calibratedImage * out_size;

    out_.convertTo(out_, CV_8UC3);
    Mat img_out = min(max(out_, 0), out_size);
    Mat img_out_rgb;
    cv::cvtColor(img_out, imageoutMat, COLOR_RGB2BGR);
    return 1;
}
int main() {
    TYPECHART chartType = TYPECHART(0);
    Mat image = cv::imread("D:\\images\\colorChecker\\7.bmp", cv::ImreadModes::IMREAD_COLOR);
    cv::imwrite("D:\\images\\colorChecker\\11.bmp", image);
    //Mat imageRotated;
    //cv::rotate(image, imageRotated, ROTATE_180);
    //image = imageRotated;
    Ptr<CCheckerDetector> detector = CCheckerDetector::create();
    if (!detector->process(image, chartType, 1))
    {
        printf("ChartColor not detected \n");
    }
    else
    {

        // get checker
        std::vector<Ptr<mcc::CChecker>> checkers = detector->getListColorChecker();

        for (Ptr<mcc::CChecker> checker : checkers)
        {
            // current checker
            Ptr<CCheckerDraw> cdraw = CCheckerDraw::create(checker);
            cdraw->draw(image);
            Mat chartsRGB = checker->getChartsRGB();
            Mat src = chartsRGB.col(1).clone().reshape(1, chartsRGB.rows / 3);
            
            src /= 255.0;

            Mat src1 = chartsRGB.col(1).clone().reshape(3, chartsRGB.rows / 3);

            src1 /= 255.0;
            Mat trgFloat = getCheckColor()/255.0;
            cv::Mat ccm1;
            std::cout << "src " << src << std::endl;
            std::cout << "trg " << trgFloat << std::endl;
            cv::solve(src,trgFloat , ccm1, DECOMP_SVD);
            //cv::Mat ccm1 = (trgFloat.t() * trgFloat).inv() * trgFloat.t()*src;
            std::cout << "ccm1 " << ccm1 << std::endl;
            cv::ccm::ColorCorrectionModel model1(src1, cv::ccm::CONST_COLOR::COLORCHECKER_Macbeth);
            model1.setCCM_TYPE(cv::ccm::CCM_3x3);
            model1.setColorSpace(cv::ccm::COLOR_SPACE_sRGB);
            //model1.setLinear()
            //model1.setLinear(cv::ccm::LINEARIZATION_IDENTITY);
            model1.run();
            Mat ccm = model1.getCCM();
            std::cout << "total " << ccm.type() << std::endl;
            for (int i = 0; i < ccm.rows;i++) {
                double sumRow = ccm.at<double>(Point(0, i))+ ccm.at<double>(Point(1, i))+ ccm.at<double>(Point(2, i));
                std::cout << "total " << sumRow << std::endl;
                for (int j = 0; j < ccm.cols; j++) {
                    //ccm.at<double>(Point(j, i)) = ccm.at<double>(Point(j, i)) / sumRow;
                }
            }
            
            std::cout << "ccm " << ccm << std::endl;
            auto loss = model1.getWeights();
            std::cout << "loss " << loss << std::endl;

            Mat img_;
            cv::cvtColor(image, img_, COLOR_BGR2RGB);
            //cv::cvtColor(image, cieImage, COLOR_BGR2Lab);
            img_.convertTo(img_, CV_64F);
            const int inp_size = 255;
            const int out_size = 255;
            img_ = img_ / inp_size;
            cv::Mat cam1Reshaped = img_.reshape(1, img_.size().height * img_.size().width);
            //Mat calibratedImage = model1.infer(img_);
            Mat calibratedImage = cam1Reshaped * ccm1.reshape(0, 3);
            calibratedImage = calibratedImage.reshape(3, img_.size().height);
           
           /*Mat calibratedImage = Mat(img_.rows, img_.cols, CV_64FC3);
            for (int i = 0; i < calibratedImage.rows; i++) {


                for (int j = 0; j < calibratedImage.cols; j++) {
                    cv::Vec<double, 3> bgrPixel = calibratedImage.at<cv::Vec<double, 3>>(i, j);
                    bgrPixel[0] = (ccm * bgrPixel).a.at<Vec<double, 3>>(0)[0];
                    bgrPixel[1] = (ccm * bgrPixel).a.at<Vec<double, 3>>(0)[1];
                    bgrPixel[2] = (ccm * bgrPixel).a.at<Vec<double, 3>>(0)[2];
                }
            }*/

            Mat out_ = calibratedImage * out_size;

            out_.convertTo(out_, CV_8UC3);
            Mat img_out = min(max(out_, 0), out_size);
            Mat out_img;
            cv::cvtColor(img_out, out_img, COLOR_RGB2BGR);
            imshow("image calibrated", out_img);
        }
    }
    imshow("image result | q or esc to quit", image);
    imshow("original", image);
    
    char key = (char)waitKey(-1);
    if (key == 27)
        return 0;
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

