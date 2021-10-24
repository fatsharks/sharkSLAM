#ifndef _SHARKS_ORBEXTRACTOR_H_
#define _SHARKS_ORBEXTRACTOR_H_

#include <opencv2/opencv.hpp>
#include <vector>
#include <list>

namespace sharkSLAM
{
    class extractorNode
    {
    public:
        void divideNode(extractorNode& n1, extractorNode& n2, extractorNode& n3, extractorNode& n4);
        std::vector<cv::KeyPoint> vKeys;
        cv::Point2i UL, UR, BL, BR;
        std::list<extractorNode>::iterator iter;
        bool bNoMore;
        extractorNode() : bNoMore(false){}
    };
    
    class ORBExtractor
    {
    protected:
        void computePyramid(cv::Mat image);
        void computeKeyPointsOctTree(std::vector<std::vector<cv::KeyPoint>>& allKeyPoints);
        std::vector<cv::KeyPoint> distributeOctTree(const std::vector<cv::KeyPoint>& vToDistributeKeys, const int& minX, const int& maxX, const int& minY, const int& maxY, const int& nFeatures, const int& level);
        void computeKeyPointsOld(std::vector<std::vector<cv::KeyPoint>>& allKeyPoints);

        std::vector<cv::Point> pattern;

        int nFeatures;
        double scaleFactor;
        int nLevels;
        int iniThFAST;
        int minThFAST;

        std::vector<int> mnFeaturesPerLevel;
        std::vector<int> umax;

        std::vector<float> mvScaleFactor;
        std::vector<float> mvInvScaleFactor;
        std::vector<float> mvLevelSigma2;
        std::vector<float> mvInvLevelSigma2;

    public:
        enum{HARRIS_SCORE = 0, FAST_SCORE = 1};
        ORBExtractor(int nFeatures, float scaleFactor, int nLevels, int iniThFAST, int minThFAST);

        void operator()(cv::InputArray image, cv::InputArray mask, std::vector<cv::KeyPoint>& keyPoints, cv::OutputArray descriptors);

        int inline getLevels(){
            return nLevels;
        }
        
        float inline getScaleFactor(){
            return scaleFactor;
        }

        std::vector<float> inline getScaleFactors(){
            return mvScaleFactor;
        }

        std::vector<float> inline getInverseScaleFactors(){
            return mvInvScaleFactor;
        }

        std::vector<float> inline getScaleSigmaASquares(){
            return mvLevelSigma2;
        }

        std::vector<float> inline getInverseScaleSigmaSquares(){
            return mvInvLevelSigma2;
        }

        std::vector<cv::Mat> mvImagePyramid;
    };
    
}
#endif