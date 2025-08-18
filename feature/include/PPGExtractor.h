/**
 * @file PPGExtractor.h
 * @brief Point-Point-Line Graph (PPG) feature extractor for SLAM
 */

#pragma once

// ==================== SYSTEM INCLUDES ====================
#include <string>
#include <vector>

// ==================== THIRD-PARTY INCLUDES ====================
#include <torch/torch.h>
#include <torch/script.h> 
#include <opencv2/opencv.hpp>
#include <Eigen/Eigen>

// ==================== LOCAL INCLUDES ====================
#include "GeometricCamera.h"
#include "PPGGraph.h"

// ==================== TYPE DEFINITIONS ====================
typedef at::TensorAccessor<float, 2UL, at::DefaultPtrTraits, signed long> TensorAccessor2D;
typedef at::TensorAccessor<float, 4UL, at::DefaultPtrTraits, signed long> TensorAccessor4D;

// ==================== FORWARD DECLARATIONS ====================
class KeyEdge;
class KeyPointEx;

/**
 * @class PPGExtractor
 * @brief Deep learning-based point and line feature extractor using PyTorch
 */
class PPGExtractor
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    // ==================== CONSTRUCTORS/DESTRUCTORS ====================

    /// Constructor with model paths and camera
    PPGExtractor(GeometricCamera *pCam, std::string dataPath);
    
    /// Destructor
    ~PPGExtractor();

    // ==================== MAIN INTERFACE ====================
    
    /// Main extraction function that processes an image and returns features
    void run(cv::Mat srcMat, std::vector<KeyPointEx>& _keypoints, std::vector<KeyPointEx>& _keypoints_un, 
             std::vector<KeyEdge>& _keyedges, cv::Mat &_descriptors);

    // ==================== PROCESSING PIPELINE ====================
    
    /// Performs neural network inference on undistorted image
    void inference(cv::Mat undistortedMat);

    /// Detects junction points from heat map
    void detectKeyPoint();

    /// Detects line segments from heat map
    void detectLines();

    /// Generates point descriptors specifically
    void genPointDescriptor();

    // ==================== UTILITY FUNCTIONS ====================
    
    /// Displays tensor for debugging purposes
    void showTensor(const torch::Tensor& ts);

    // ==================== ACCESSOR FUNCTIONS ====================
    
    /// Get detected keypoints
    const std::vector<KeyPointEx>& getKeyPoints() const { return mvKeyPoints; }
    
    /// Get detected line segments
    const std::vector<KeyEdge>& getKeyEdges() const { return mvKeyEdges; }

private:
    // ==================== PRIVATE HELPER FUNCTIONS ====================
    
    /// Refines heat map using adaptive thresholding
    void refineHeatMap(torch::Tensor &scoreMap);

    /// Performs bilinear interpolation on matrix
    float bilinearInterpolation(const Eigen::MatrixXf & M, float ptX, float ptY); 

    /// Computes connected line score for junction optimization
    float connectedLineScore(unsigned int pid, float biax=0., float biay=0.);

    /// Computes heat map score along line segment
    float heatMapLineScore(const Eigen::Vector2f &ps, const Eigen::Vector2f &pe);

    /// Computes inlier rate along line segment
    float heatMapInlierRate(const Eigen::Vector2f &ps, const Eigen::Vector2f &pe);

public:
    // ==================== EXTRACTED FEATURES ====================
    std::vector<KeyPointEx> mvKeyPoints;    ///< Detected junction points
    std::vector<KeyEdge> mvKeyEdges;        ///< Detected line segments
    torch::Tensor normDesc;                 ///< Normalized descriptors

    // ==================== CONFIGURATION PARAMETERS ====================
    static int          DESC_DIM_SIZE;            ///< Descriptor dimension size
    static float        JUNCTION_THRESH;          ///< Junction detection threshold
    static int          JUNCTION_NMS_RADIUS;      ///< Non-maximum suppression radius for junctions
    static unsigned int JUNCTION_MAX_NUM;         ///< Maximum number of junctions to detect
    static float        LINE_VALID_THRESH;        ///< Line validation threshold
    static float        LINE_VALID_RATIO;         ///< Minimum ratio of valid pixels for line detection
    static float        LINE_DISTTHRESH;          ///< Distance threshold for line overlap detection
    static int          HEATMAP_REFINE_SZ;        ///< Grid size for heat map refinement
    static float        LINE_HEATMAP_THRESH;      ///< Heat map threshold for line validation
    static float        LINE_INLIER_RATE;         ///< Required inlier rate for line segments

private:
    // ==================== NEURAL NETWORK MODELS ====================
    torch::jit::Module model_backbone;      ///< Backbone feature extraction network
    torch::jit::Module model_descriptor;    ///< Descriptor generation network
    torch::jit::Module model_heatmap;       ///< Line heat map generation network
    torch::jit::Module model_junction;      ///< Junction detection network

    // ==================== PROCESSING TENSORS ====================
    torch::Tensor input_tensor;             ///< Input image tensor
    torch::Tensor featureMap;               ///< Feature map from backbone
    torch::Tensor junctions;                ///< Junction heat map
    torch::Tensor heatmap;                  ///< Line heat map
    torch::Tensor descriptors;              ///< Raw descriptors
    torch::Tensor junc_pred;                ///< Processed junction predictions
    torch::Tensor heatmap_score;            ///< Processed heat map scores

    // ==================== CAMERA PARAMETERS ====================
    bool mbFisheye;                         ///< Flag indicating fisheye camera model
    cv::Mat mK;                             ///< Camera intrinsic matrix
    cv::Mat mD;                             ///< Camera distortion coefficients
    cv::Mat mX, mY;                         ///< Undistortion maps
    int mnImHeight, mnImWidth;              ///< Image dimensions
    float invScale;                         ///< Inverse scale factor

    // ==================== PROCESSING DATA ====================
    cv::Mat mImg;                           ///< Current input image
    cv::Mat heatMat;                        ///< Heat map as OpenCV matrix
    Eigen::MatrixXf eigenHeat;              ///< Heat map as Eigen matrix (undistorted)
    unsigned char *nmsFlag;                 ///< Non-maximum suppression flags
    
    // ==================== STATIC CONFIGURATION ====================
    static torch::Device dev;               ///< PyTorch device (CPU/GPU)
};
