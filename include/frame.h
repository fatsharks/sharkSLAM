#ifndef _SHARKS_FRAME_H_
#define _SHARKS_FRAME_H_

#include <opencv2/opencv.hpp>
#include <vector>
#include <DBoW3/DBoW3.h>
#include "mapPoint.h"
#include "ORBExtractor.h"

namespace sharkSLAM
{
#define FRAME_GRID_ROWS 48
#define FRAME_GRID_COLS 64

    class Frame
    {
    private:
    public:
        Frame();
        Frame(const Frame &frame);
        /**
         * @brief 为RGBD相机准备的帧构造函数
         * 
         * @param[in] imGray        对RGB图像灰度化之后得到的灰度图像
         * @param[in] imDepth       深度图像
         * @param[in] timeStamp     时间戳
         * @param[in] extractor     特征点提取器句柄
         * @param[in] voc           ORB特征点词典的句柄
         * @param[in] K             相机的内参数矩阵
         * @param[in] distCoef      相机的去畸变参数
         * @param[in] bf            baseline*bf
         * @param[in] thDepth       远点和近点的深度区分阈值
         */
        Frame(const cv::Mat &imGray, const cv::Mat &imDepth, const double &timeStamp, ORBExtractor *extractor, DBoW3::Vocabulary *voc, cv::Mat &K, cv::Mat &distCoef, const float &bf, const float &threshold);

        /**
         * @brief 提取图像的ORB特征，提取的关键点存放在mvKeys，描述子存放在mDescriptors
         * 
         * @param[in] flag          标记是左图还是右图。0：左图  1：右图
         * @param[in] im            等待提取特征点的图像
         */
        void extractORB(int flag, const cv::Mat &im);

        // 存放在mBowVec中
        /**
         * @brief 计算词袋模型 
         * @details 计算词包 mBowVec 和 mFeatVec ，其中 mFeatVec 记录了属于第i个node（在第4层）的ni个描述子
         * @see CreateInitialMapMonocular() TrackReferenceKeyFrame() Relocalization()
         */
        void computeBoW();

        // 用Tw2c更新mTw2c
        /**
         * @brief 用 Tw2c 更新 mTw2c 以及类中存储的一系列位姿
         * 
         * @param[in] Tw2c 从世界坐标系到当前帧相机位姿的变换矩阵
         */
        void setPose(cv::Mat tw2c);

        /**
         * @brief 根据相机位姿,计算相机的旋转,平移和相机中心等矩阵.
         * @details 其实就是根据Tcw计算mRcw、mtcw和mRwc、mOw.
         */
        void updatePoseMatrixes();

        /**
         * @brief 返回位于当前帧位姿时,相机的中心
         * 
         * @return cv::Mat 相机中心在世界坐标系下的3D点坐标
         */
        inline cv::Mat getCameraCenter(){ return mOw.clone(); }

        /**
         * @brief Get the Rotation Inverse object
         * mRwc存储的是从当前相机坐标系到世界坐标系所进行的旋转，而我们一般用的旋转则说的是从世界坐标系到当前相机坐标系的旋转
         * @return 返回从当前帧坐标系到世界坐标系的旋转
         */
        inline cv::Mat getRotationInverse(){ return mRc2w.clone(); }
        /**
         * @brief 判断路标点是否在视野中
         * 步骤
         * Step 1 获得这个地图点的世界坐标
         * Step 2 关卡一：检查这个地图点在当前帧的相机坐标系下，是否有正的深度.如果是负的，表示出错，返回false
         * Step 3 关卡二：将MapPoint投影到当前帧的像素坐标(u,v), 并判断是否在图像有效范围内
         * Step 4 关卡三：计算MapPoint到相机中心的距离, 并判断是否在尺度变化的距离内
         * Step 5 关卡四：计算当前视角和“法线”夹角的余弦值, 若小于设定阈值，返回false
         * Step 6 根据地图点到光心的距离来预测一个尺度（仿照特征点金字塔层级）
         * Step 7 记录计算得到的一些参数
         * @param[in] pMP                       当前地图点
         * @param[in] viewingCosLimit           夹角余弦，用于限制地图点和光心连线和法线的夹角
         * @return true                         地图点合格，且在视野内
         * @return false                        地图点不合格，抛弃
         */
        bool isInFrustum(MapPoint* pMP, float viewingCosLimit);
        ~Frame();

    public:
        ///用于重定位的ORB特征字典
        DBoW3::Vocabulary* mpVocabulary;

        ///ORB特征提取器句柄,其中右侧的提取器句柄只会在双目输入的情况中才会被用到
        ORBExtractor* mpORBextractorLeft, *mpORBextractorRight;

        double mTimeStamp;
        ///相机的内参数矩阵
        cv::Mat mK;

        static float fx;        ///<x轴方向焦距
        static float fy;        ///<y轴方向焦距
        static float cx;        ///<x轴方向光心偏移
        static float cy;        ///<y轴方向光心偏移
        static float invfx;     ///<x轴方向焦距的逆
        static float invfy;     ///<x轴方向焦距的逆

        ///去畸变参数
        cv::Mat mDistCoef;

        float mbf;
        float mb;

        ///判断远点和近点的深度阈值
        float mThresholdDepth;
        // Number of KeyPoints.
        int N;

        ///原始左图像提取出的特征点（未校正）
        std::vector<cv::KeyPoint> mvKeysLeft;
        ///原始右图像提取出的特征点（未校正）
        std::vector<cv::KeyPoint> mvKeysRight;
        ///校正mvKeys后的特征点
        std::vector<cv::KeyPoint> mvKeysUn;

    private:
        // Rotation, translation and camera center
        cv::Mat mRw2c; ///< Rotation from world to camera
        cv::Mat mtw2c; ///< Translation from world to camera
        cv::Mat mRc2w; ///< Rotation from camera to world
        cv::Mat mOw;  ///< mtwc,Translation from camera to world
    };

    class KeyFrame;

}
#endif