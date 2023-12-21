#include <iostream>
#include <fstream>
#include <string>
//#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>


void SpeckleFileter1f(cv::Mat1f& src, int areaThred, float thredP, float minThred, float maxThred)
{
    int  h = src.rows;
    int  w = src.cols;
    int  pixels = h * w;

    bool* hasLabeled = new bool[pixels];
    int* connectX = new int[pixels];
    int* connectY = new int[pixels];
    int* recordLoc = new int[pixels];

    float* srcPtr = (float*)src.data;

    int hsize = w;
    int loci = 0;
    int h1 = h - 1;
    int w1 = w - 1;

    for (int k = 0; k < pixels; k++)
    {
        hasLabeled[k] = false;
    }

    for (int i = 0; i < h; i++)
    {
        for (int j = 0; j < w; j++)
        {
            int loc = loci + j;
            int loc3Z = loc;

            if (!hasLabeled[loc])
            {
                int connectNum = 0;
                int recordNum = 0;

                connectX[connectNum] = j;
                connectY[connectNum] = i;

                hasLabeled[loc] = true;

                while (connectNum >= 0 && srcPtr[loc3Z] > 0)
                {
                    int curX = connectX[connectNum];
                    int curY = connectY[connectNum];

                    int curLoc = curX + curY * w;
                    int curLocZ = curLoc;
                    float curValue = srcPtr[curLocZ];

//                    float ljx = powf(curValue,2);
//                    float depthThred = fmin(maxThred, fmax(minThred, thredP * powf(curValue,2)));
                    float dis2 =
                            powf(curValue, 2);
                    float depthThred = thredP * dis2;
                    depthThred = depthThred < minThred ? minThred: depthThred;
                    depthThred = depthThred > maxThred ? maxThred : depthThred;

                    recordLoc[recordNum] = curLocZ;

                    connectNum -= 1;
                    recordNum += 1;

                    //左边
                    if (curX > 1 && (!hasLabeled[curLoc - 1]) && fabs(srcPtr[curLocZ - 1] - curValue) < depthThred)
                    {
                        connectNum += 1;
                        connectX[connectNum] = curX - 1;
                        connectY[connectNum] = curY;
                        hasLabeled[curLoc - 1] = true;
                    }
                    //右边
                    if (curX < w1 && (!hasLabeled[curLoc + 1]) && fabs(srcPtr[curLocZ + 1] - curValue) < depthThred)
                    {
                        connectNum += 1;
                        connectX[connectNum] = curX + 1;
                        connectY[connectNum] = curY;
                        hasLabeled[curLoc + 1] = true;
                    }
                    //上边
                    if (curY > 1 && (!hasLabeled[curLoc - w]) && fabs(srcPtr[curLocZ - hsize] - curValue) < depthThred)
                    {
                        connectNum += 1;
                        connectX[connectNum] = curX;
                        connectY[connectNum] = curY - 1;
                        hasLabeled[curLoc - w] = true;
                    }
                    //下边
                    if (curY < h1 && (!hasLabeled[curLoc + w]) && fabs(srcPtr[curLocZ + hsize] - curValue) < depthThred)
                    {
                        connectNum += 1;
                        connectX[connectNum] = curX;
                        connectY[connectNum] = curY + 1;
                        hasLabeled[curLoc + w] = true;
                    }

                }

                //判断面积
                if (recordNum < areaThred && srcPtr[loc3Z] > 0)
                {
                    for (int k = 0; k < recordNum; k++)
                    {
                        srcPtr[recordLoc[k]] = 0;
                    }
                }
            }
        }

        loci += w;
    }

    delete[] hasLabeled;
    delete[] connectX;
    delete[] connectY;
    delete[] recordLoc;
}
cv::Mat Denoise(cv::Mat depthImg)
{
    cv::Mat1f floatImg;

    cv::Mat mask = depthImg > 3.5 * 1000.0;
    depthImg.setTo(0, mask);

    depthImg.convertTo(floatImg, CV_32FC1);
    floatImg /= 1000.0;

    SpeckleFileter1f(floatImg, 100, 0.01, 0.003, 0.02);

    floatImg *= 1000.0;
    cv::Mat int16Img;
    floatImg.convertTo(int16Img, CV_16UC1);

    return int16Img;
}

int main()
{
    std::string imageFileName = "/home/indemind/Code/PycharmProjects/Depth_Estimation/depth_estimation_test/result_PC/result_SLAM_Euroc_1.2-3_0717_160m01_D10.7.0_new/disp_scaleX256_uint16/filst.txt";
    std::string outputDir = "depthDenoise/";
    std::ifstream ifs;
    std::vector<std::string> file_names;
    ifs.open(imageFileName);
    if (!ifs.is_open())
    {
        std::cout << "open file!" << std::endl;
        return 0;
    }
    std::string fileStr;
    while (std::getline(ifs, fileStr))
    {
        file_names.push_back(fileStr);
    }

    for (int i = 0; i < file_names.size(); ++i)
    {
        std::string imageName = file_names[i];
        cv::Mat imageDisp = cv::imread(file_names[i],-1);
        cv::Mat imageDepth;
//        std::cout << imageDisp << std::endl;
        imageDisp = imageDisp /256.0;
//        std::cout << imageDisp << std::endl;
        imageDisp.convertTo(imageDisp, CV_32FC1);
        imageDepth = 14.2 / imageDisp;
//        std::cout << imageDepth << std::endl;
        cv::Mat imageDepthCopy = imageDepth.clone();
        imageDepth = imageDepth * 1000;
//        std::cout << imageDepth.channels() <<std::endl;

//        cv::cvtColor(imageDepth, imageDepth, CV_BGR2GRAY);
        imageDepth.convertTo(imageDepth, CV_16UC1);
//        std::cout << imageDepth.channels() << std::endl;
        cv::Mat imageDenoise = Denoise(imageDepth);
        int idx = imageName.find_last_of("/");
        std::string imageNameLast = imageName.substr(idx + 1, imageName.length() - idx);
//        std::cout << imageName << std::endl;
        std::cout << imageNameLast << std::endl;
        cv::imwrite(outputDir + imageNameLast, imageDenoise);
        cv::imwrite(outputDir + imageNameLast + ".png", imageDepthCopy);
    }

    std::cout << "Hello, World!" << std::endl;
    return 0;
}
