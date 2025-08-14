/**
 * @file PPGExtractor.h
 * @brief Point-Point-Line Graph (PPG) feature extractor for SLAM
 * @details This class implements a deep learning-based feature extractor that detects
 *          junctions (keypoints), line segments, and generates descriptors using
 *          PyTorch models. It combines point and line features for robust visual SLAM.
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
 * @brief Deep learning-based point and line feature extractor
 * @details This class uses PyTorch neural networks to extract junction points and line segments
 *          from images. It performs the following operations:
 *          - Junction detection using heat map analysis
 *          - Line segment detection and optimization
 *          - Feature descriptor generation
 *          - Camera distortion handling and feature undistortion
 */

class PPGExtractor
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    // ==================== CONSTRUCTORS/DESTRUCTORS ====================
    
    /**
     * @brief Constructor that initializes the PPG extractor with camera and model parameters
     * @param pCam Pointer to geometric camera for distortion correction
     * @param dataPath Path to the directory containing PyTorch model files
     */
    PPGExtractor(GeometricCamera *pCam, std::string dataPath);
    
    /**
     * @brief Destructor
     */
    ~PPGExtractor();

    // ==================== MAIN INTERFACE ====================
    
    /**
     * @brief Main extraction function that processes an image and returns features
     * @param srcMat Input grayscale image
     * @param _keypoints Output vector of detected keypoints (distorted)
     * @param _keypoints_un Output vector of undistorted keypoints
     * @param _keyedges Output vector of detected line segments
     * @param _descriptors Output matrix of feature descriptors
     */
    void run(cv::Mat srcMat, std::vector<KeyPointEx>& _keypoints, std::vector<KeyPointEx>& _keypoints_un, 
             std::vector<KeyEdge>& _keyedges, cv::Mat &_descriptors);

    // ==================== PROCESSING PIPELINE ====================
    
    /**
     * @brief Performs neural network inference on undistorted image
     * @param undistortedMat Undistorted input image
     */
    void inference(cv::Mat undistortedMat);

    /**
     * @brief Detects junction points from heat map
     */
    void detectKeyPoint();

    /**
     * @brief Detects line segments from heat map
     */
    void detectLines();

    /**
     * @brief Optimizes junction positions using iterative refinement
     */
    void optimizeJunctions();

    /**
     * @brief Generates descriptors for detected features
     */
    void genDescriptor();

    /**
     * @brief Generates point descriptors specifically
     */
    void genPointDescriptor();

    // ==================== UTILITY FUNCTIONS ====================
    
    /**
     * @brief Displays tensor for debugging purposes
     * @param ts Tensor to display
     */
    void showTensor(const torch::Tensor& ts);

    // ==================== ACCESSOR FUNCTIONS ====================
    
    /**
     * @brief Get detected keypoints
     * @return Vector of detected keypoints
     */
    const std::vector<KeyPointEx>& getKeyPoints() const { return mvKeyPoints; }
    
    /**
     * @brief Get detected line segments
     * @return Vector of detected line segments
     */
    const std::vector<KeyEdge>& getKeyEdges() const { return mvKeyEdges; }

private:
    // ==================== PRIVATE HELPER FUNCTIONS ====================
    
    /**
     * @brief Refines heat map using adaptive thresholding
     * @param scoreMap Input/output score map tensor
     */
    void refineHeatMap(torch::Tensor &scoreMap);

    /**
     * @brief Performs bilinear interpolation on matrix
     * @param M Input matrix
     * @param ptX X coordinate (can be fractional)
     * @param ptY Y coordinate (can be fractional)
     * @return Interpolated value
     */
    float bilinearInterpolation(const Eigen::MatrixXf & M, float ptX, float ptY); 

    /**
     * @brief Computes connected line score for junction optimization
     * @param pid Point ID
     * @param biax X bias for optimization
     * @param biay Y bias for optimization
     * @return Score value
     */
    float connectedLineScore(unsigned int pid, float biax=0., float biay=0.);

    /**
     * @brief Computes heat map score along line segment
     * @param ps Start point of line segment
     * @param pe End point of line segment
     * @return Line score from heat map
     */
    float heatMapLineScore(const Eigen::Vector2f &ps, const Eigen::Vector2f &pe);

    /**
     * @brief Computes inlier rate along line segment
     * @param ps Start point of line segment
     * @param pe End point of line segment
     * @return Inlier rate (0.0 to 1.0)
     */
    float heatMapInlierRate(const Eigen::Vector2f &ps, const Eigen::Vector2f &pe);

    // ==================== MEMBER VARIABLES ====================
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
    static int          OPTIMIZE_ITER_NUM;        ///< Number of optimization iterations
    static float        OPTIMIZE_ITER_DECAY;     ///< Step size decay factor for optimization

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

    // ==================== DEPRECATED/EXPERIMENTAL ====================
    torch::jit::Module model_1;             ///< Experimental model (unused)
    torch::jit::Module model_2;             ///< Experimental model (unused)
    torch::Tensor featureMap2;              ///< Experimental feature map (unused)
    
    // ==================== STATIC CONFIGURATION ====================
    static torch::Device dev;               ///< PyTorch device (CPU/GPU)
};
