#ifndef _SHARKS_MATCHER_H_
#define _SHARKS_MATCHER_H_

#include <vector>
#include <opencv2/opencv.hpp>
#include "frame.h"
#include "mapPoint.h"

namespace sharkSLAM
{
    class ORBMatcher
    {
    protected:
        /**
         * @brief 检查极线距离
         * @param[in] kp1   特征点1
         * @param[in] kp2   特征点2
         * @param[in] F12   两帧之间的基础矩阵
         * @param[in] pKF   //? 关键帧?
         * @return true 
         * @return false 
         **/
        bool checkDistEpipolarLine(const cv::KeyPoint &kp1, const cv::KeyPoint &kp2, const cv::Mat &F12, const KeyFrame *pKF);

    /**
     * @brief 根据观察的视角来计算匹配的时的搜索窗口大小
     * @param[in] viewCos   视角的余弦值
     * @return float        搜索窗口的大小
     */
        float radiusByViewingCos(const float &viewCos);

    /**
     * @brief 筛选出在旋转角度差落在在直方图区间内数量最多的前三个bin的索引
     * 
     * @param[in] histo         匹配特征点对旋转方向差直方图
     * @param[in] L             直方图尺寸
     * @param[in & out] ind1          bin值第一大对应的索引
     * @param[in & out] ind2          bin值第二大对应的索引
     * @param[in & out] ind3          bin值第三大对应的索引
     */
        void computerThreeMaxima(std::vector<int> *histo, const int L, int &ind1, int &ind2, int &ind3);

        float mfNNratio;    //  最优评分和次优评分的比例
        bool mbCheckOritentation;// 是否检查特征点方向

    public:
        // 阈值
        static const int TH_LOW;
        static const int TH_HIGH;
        static const int HISTO_LENGTH;

    public:
        /**
     * Constructor
     * @param nnratio  ratio of the best and the second score   最优和次优评分的比例
     * @param checkOri check orientation                        是否检查方向
     */
        ORBMatcher(float nnratio = 0.6, bool checkori = true);

    /**
     * @brief Computes the Hamming distance between two ORB descriptors 计算地图点和候选投影点的描述子距离
     * @param[in] a     一个描述子
     * @param[in] b     另外一个描述子
     * @return int      描述子的汉明距离
     */
        static int descriptionDistance(const cv::Mat &descriptor1, const cv::Mat &descriptor2);

    // Search matches between Frame keypoints and projected MapPoints. Returns number of matches
    // Used to track the local map (Tracking)
    /**
     * @brief 通过投影地图点到当前帧，对Local MapPoint进行跟踪
     * 步骤
     * Step 1 遍历有效的局部地图点
     * Step 2 设定搜索搜索窗口的大小。取决于视角, 若当前视角和平均视角夹角较小时, r取一个较小的值
     * Step 3 通过投影点以及搜索窗口和预测的尺度进行搜索, 找出搜索半径内的候选匹配点索引
     * Step 4 寻找候选匹配点中的最佳和次佳匹配点
     * Step 5 筛选最佳匹配点
     * @param[in] F                         当前帧
     * @param[in] vpMapPoints               局部地图点，来自局部关键帧
     * @param[in] th                        搜索范围
     * @return int                          成功匹配的数目
     */
        int searchByProjection(Frame &frame, const std::vector<MapPoint *> &vpMapPoints, const float threshold = 3);

    // Project MapPoints tracked in last frame into the current frame and search matches.
    // Used to track from previous frame (Tracking)
    /**
     * @brief 将上一帧跟踪的地图点投影到当前帧，并且搜索匹配点。用于跟踪前一帧
     * 步骤
     * Step 1 建立旋转直方图，用于检测旋转一致性
     * Step 2 计算当前帧和前一帧的平移向量
     * Step 3 对于前一帧的每一个地图点，通过相机投影模型，得到投影到当前帧的像素坐标
     * Step 4 根据相机的前后前进方向来判断搜索尺度范围
     * Step 5 遍历候选匹配点，寻找距离最小的最佳匹配点 
     * Step 6 计算匹配点旋转角度差所在的直方图
     * Step 7 进行旋转一致检测，剔除不一致的匹配
     * @param[in] CurrentFrame          当前帧
     * @param[in] LastFrame             上一帧
     * @param[in] th                    搜索范围阈值，默认单目为7，双目15
     * @param[in] bMono                 是否为单目
     * @return int                      成功匹配的数量
     */
        int searchByProjection(Frame &curFrame, const Frame &lastFrame, const float threshold, const bool bMono);

    // Project MapPoints seen in KeyFrame into the Frame and search matches.
    // Used in relocalisation (Tracking)
    /**
     * @brief 通过投影的方式将关键帧中的地图点投影到当前帧中,并且进行匹配
     * 
     * @param[in] CurrentFrame      当前帧
     * @param[in] pKF               关键帧
     * @param[in] sAlreadyFound     已经寻找得到的地图点
     * @param[in] th                //窗口大小的阈值
     * @param[in] ORBdist           //描述子最小距离阈值
     * @return int                  //匹配到的点的数目
     */
        int searchByProjection(Frame &curFrame, KeyFrame *pKF, const std::set<MapPoint *> &sAlreadyFound, const float threshold, const int ORBDist);

    // Project MapPoints using a Similarity Transformation and search matches.
    // Used in loop detection (Loop Closing)
    /**
     * @brief 根据Sim3变换，将每个vpPoints投影到pKF上，并根据尺度确定一个搜索区域，
     * @detials 根据该MapPoint的描述子与该区域内的特征点进行匹配，如果匹配误差小于TH_LOW即匹配成功，更新vpMatched
     * @param[in] pKF               要投影到的关键帧
     * @param[in] Scw               相似变换
     * @param[in] vpPoints          空间点
     * @param[in] vpMatched         已经得到的空间点和关键帧上点的匹配关系
     * @param[in] th                搜索窗口的阈值
     * @return int                  匹配的特征点数目
     */
        int searchByProjection(KeyFrame *pKF, cv::Mat scw, const std::vector<MapPoint *> &vpPoints, std::vector<MapPoint *> &vpMatched, int threshold);

    // Search matches between MapPoints in a KeyFrame and ORB in a Frame.
    // Brute force constrained to ORB that belong to the same vocabulary node (at a certain level)
    // Used in Relocalisation and Loop Detection
    /*
    * @brief 通过词袋，对关键帧的特征点进行跟踪
    * 步骤
    * Step 1：分别取出属于同一node的ORB特征点(只有属于同一node，才有可能是匹配点)
    * Step 2：遍历KF中属于该node的特征点
    * Step 3：遍历F中属于该node的特征点，寻找最佳匹配点
    * Step 4：根据阈值 和 角度投票剔除误匹配
    * Step 5：根据方向剔除误匹配的点
    * @param  pKF               KeyFrame
    * @param  F                 Current Frame
    * @param  vpMapPointMatches F中MapPoints对应的匹配，NULL表示未匹配
    * @return                   成功匹配的数量
    */
        int searchByBow(KeyFrame *pKF, Frame &frame, std::vector<MapPoint *> &vpMapPointMatchers);
        int searchByBow(KeyFrame *pKF1, KeyFrame *pKF2, std::vector<MapPoint *> &vpMatchers12);

    // Matching for the Map Initialization (only used in the monocular case)
    /**
     * @brief 单目初始化中用于参考帧和当前帧的特征点匹配
     * 步骤
     * Step 1 构建旋转直方图
     * Step 2 在半径窗口内搜索当前帧F2中所有的候选匹配特征点 
     * Step 3 遍历搜索搜索窗口中的所有潜在的匹配候选点，找到最优的和次优的
     * Step 4 对最优次优结果进行检查，满足阈值、最优/次优比例，删除重复匹配
     * Step 5 计算匹配点旋转角度差所在的直方图
     * Step 6 筛除旋转直方图中“非主流”部分
     * Step 7 将最后通过筛选的匹配好的特征点保存
     * @param[in] F1                        初始化参考帧                  
     * @param[in] F2                        当前帧
     * @param[in & out] vbPrevMatched       本来存储的是参考帧的所有特征点坐标，该函数更新为匹配好的当前帧的特征点坐标
     * @param[in & out] vnMatches12         保存参考帧F1中特征点是否匹配上，index保存是F1对应特征点索引，值保存的是匹配好的F2特征点索引
     * @param[in] windowSize                搜索窗口
     * @return int                          返回成功匹配的特征点数目
     */
        int searchForInitialization(Frame &frame1, Frame &frame2, std::vector<cv::Point2f> &vbPrevMatched, std::vector<int> &vnMatches12, int windowSize = 10);

    // Matching to triangulate new MapPoints. Check Epipolar Constraint.
    /**
     * @brief 利用基本矩阵F12，在两个关键帧之间未匹配的特征点中产生新的3d点
     * @param pKF1          关键帧1
     * @param pKF2          关键帧2
     * @param F12           基础矩阵
     * @param vMatchedPairs 存储匹配特征点对，特征点用其在关键帧中的索引表示，下标是关键帧1的特征点id，存储的是关键帧2的特征点id
     * @param bOnlyStereo   在双目和rgbd情况下，是否要求特征点在右图存在匹配
     * @return              成功匹配的数量
     */
        int searchForTriangulation(KeyFrame *pKF1, KeyFrame *pKF2, cv::Mat F12, std::vector<std::pair<size_t, size_t>> &vMatchedPairs, const bool bOnlyStereo);

    /**
     * @brief 通过Sim3变换，确定pKF1的特征点在pKF2中的大致区域，同理，确定pKF2的特征点在pKF1中的大致区域
     * @detials 在该区域内通过描述子进行匹配捕获pKF1和pKF2之前漏匹配的特征点，更新vpMatches12（之前使用SearchByBoW进行特征点匹配时会有漏匹配）
     * @param[in] pKF1              关键帧1
     * @param[in] pKF2              关键帧2
     * @param[in] vpMatches12       两帧特征点的匹配关系
     * @param[in] s12               缩放因子,SIM3中的吧
     * @param[in] R12 
     * @param[in] t12 
     * @param[in] th                搜索窗口阈值
     * @return int                  匹配到的点的个数
     */
    
        int searchBySim3(KeyFrame *pKF1, KeyFrame *pKF2, std::vector<MapPoint *> &vpMatches12, const float &s12, const cv::Mat &R12, const cv::Mat &t12, const float threshold);

    // Project MapPoints into KeyFrame and search for duplicated MapPoints.
    /**
     * @brief 将地图点投影到关键帧中进行匹配和融合;并且地图点的替换可以在这个函数中进行
     * @param[in] pKF           关键帧
     * @param[in] vpMapPoints   地图点
     * @param[in] th            搜索窗口的阈值
     * @return int 
     */
        int fuse(KeyFrame *pKF, const std::vector<MapPoint *> &vpMapPoints, const float threshold = 3.0);

    /**
     * @brief 将地图点投影到关键帧中进行,但是由于种种原因,地图点还不能够在这个函数中完成替换操作
     * 
     * @param[in] pKF               关键帧
     * @param[in] Scw               仿射变换
     * @param[in] vpPoints          给出的地图点
     * @param[in] th                搜索窗口阈值
     * @param[out] vpReplacePoint   需要替换掉的地图点,键值对
     * @return int                  融合的地图点个数
     */
        int fuse(KeyFrame *pKF, cv::Mat scw, const std::vector<MapPoint *> &vpPoints, float threshold, std::vector<MapPoint *> &vpReplacePoint);
    };

}

#endif