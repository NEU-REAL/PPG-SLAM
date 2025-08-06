#pragma once
#include <torch/torch.h>
#include <torch/script.h> 
#include <opencv2/opencv.hpp>
#include <Eigen/Eigen>
#include <list>
#include "KannalaBrandt8.h"
#include "Pinhole.h"
#include "PPGGraph.h"

typedef at::TensorAccessor<float, 2UL, at::DefaultPtrTraits, signed long> tensorAccer2d;
typedef at::TensorAccessor<float, 4UL, at::DefaultPtrTraits, signed long> tensorAccer4d;

class KeyEdge;
class KeyPointEx;

class PPGExtractor
{
public:
    PPGExtractor(GeometricCamera *pCam, std::string dataPath);
    ~PPGExtractor();

    void run(cv::Mat srcMat, std::vector<KeyPointEx>& _keypoints, std::vector<KeyPointEx>& _keypoints_un, std::vector<KeyEdge>& _keyedges, cv::Mat &_descriptors);

    void inference(cv::Mat undistortedMat);

    void detectKeyPoint();

    void detectLines();

    void optimizeJunctions();

    void genDescriptor();

    void genPointDescriptor();

    void showTensor(const torch::Tensor& ts);

private:
    void refineHeatMap(torch::Tensor &scoreMap);

    float bilinearInterpolation(const Eigen::MatrixXf & M, float ptX, float ptY); 

    float connectedLineScore(unsigned int pid, float biax=0., float biay=0.);

    float heatMapLineScore(const Eigen::Vector2f &ps, const Eigen::Vector2f &pe);

    float heatMapInlierRate(const Eigen::Vector2f &ps, const Eigen::Vector2f &pe);

public:
    std::vector<KeyPointEx> mvKeyPoints;
    std::vector<KeyEdge> mvKeyEdges;
    torch::Tensor normDesc;

public:
    static int          DESC_DIM_SIZE;            // 描述子维度
    static float        JUNCTION_THRESH;          // 初步筛选junction的阈值
    static int          JUNCTION_NMS_RADIUS;      // nms 的搜索半径
    static unsigned int JUNCTION_MAX_NUM;		    // junction 最大数量，按照得分从大到小排序
    static float        LINE_VALID_THRESH;        // 初步筛选heatmap的阈值
    static float        LINE_VALID_RATIO;              // heatmap中线所在像素占图像比例
    static float        LINE_DISTTHRESH;          // 判断直线是否重合，论文中给定的点到直线距离3像素
    static int          HEATMAP_REFINE_SZ;        // 对heatmap均衡化，均衡化的网格大小
    static float        LINE_HEATMAP_THRESH;       // 对sample点，阈值大于此才算是直线
    static float        LINE_INLIER_RATE;         // 对所有sample点，要有rate%点被认为直线
    static int          OPTIMIZE_ITER_NUM;        // 迭代次数
    static float        OPTIMIZE_ITER_DECAY;    // 每次迭代步长衰减 

private:
// 原始数据
    cv::Mat mImg;
	torch::Tensor input_tensor;
	torch::Tensor featureMap;
	torch::Tensor junctions;
	torch::Tensor heatmap;
	torch::Tensor descriptors;
// 后处理数据
    torch::Tensor junc_pred;
    torch::Tensor heatmap_score;
    unsigned char *nmsFlag;
// param
    int mnImHeight,mnImWidth;
	float invScale;

    torch::jit::Module model_backbone;
    torch::jit::Module model_descriptor;
    torch::jit::Module model_heatmap;
    torch::jit::Module model_junction;

    torch::jit::Module model_1;
    torch::jit::Module model_2;
    torch::Tensor featureMap2;
    
    static torch::Device dev;

    bool mbFisheye;
    cv::Mat mK, mD;
    cv::Mat mX, mY;
    cv::Mat heatMat;
	Eigen::MatrixXf eigenHeat; // undistorted image
};
