// 使用cpu 分析，速度太慢了，放弃
#include "PPGExtractor.h"
#include <map>
#include <unistd.h>
#include "iostream"
#include <opencv2/core/eigen.hpp>
#include <torch/nn/functional.h>

// 以下表值受线段长度等级影响
// 每个先端采样数量
const float invSampleGapTable[4] = {0.3333, 0.200, 0.1427, 0.1111}; // 1/3 1/5 1/7 1/9 (1/pix)
// 每个点的环形采样
const int circleTableL[4] = {1,8,12,16};
const int circleTableX[4][16]={{ 0 }, { 1, 1, 1, 0,-1,-1,-1, 0},
                          { 0, 1, 2, 2, 2, 1, 0,-1,-2,-2,-2,-1},
                          { 0, 1, 2, 3, 3, 3, 2, 1, 0,-1,-2,-3,-3,-3,-2,-1}};
const int circleTableY[4][16]={{ 0 }, {-1, 0, 1, 1, 1, 0,-1,-1},
                          { 2, 2, 1, 0,-1,-2,-2,-2,-1, 0, 1, 2},
                          {-3,-3,-2,-1, 0, 1, 2, 3, 3, 3, 2, 1, 0,-1,-2,-3}};

torch::Device PPGExtractor::dev = torch::Device(torch::kCUDA,0);

int             PPGExtractor::DESC_DIM_SIZE = 256;
float           PPGExtractor::JUNCTION_THRESH = 1. / 128.;
int             PPGExtractor::JUNCTION_NMS_RADIUS = 4;
unsigned int    PPGExtractor::JUNCTION_MAX_NUM = 500;
float           PPGExtractor::LINE_VALID_THRESH = 1.e-2;
float           PPGExtractor::LINE_VALID_RATIO = 0.3;
float        	PPGExtractor::LINE_DISTTHRESH = 2.;
int             PPGExtractor::HEATMAP_REFINE_SZ = 16;
float           PPGExtractor::LINE_HEATMAP_THRESH = 0.2;
float           PPGExtractor::LINE_INLIER_RATE = 0.8;
int             PPGExtractor::OPTIMIZE_ITER_NUM = 4;
float        PPGExtractor::OPTIMIZE_ITER_DECAY = 0.6;


PPGExtractor::PPGExtractor(GeometricCamera *pCam, std::string dataPath)
{
	mK = pCam->toK();
	mD = pCam->toD();
	mnImWidth = pCam->imWidth();
	mnImHeight = pCam->imHeight();
	mbFisheye = (pCam->mnType ==pCam->CAM_FISHEYE);

	if(mbFisheye)
		cv::fisheye::initUndistortRectifyMap(mK, mD, cv::Mat::eye(3,3,CV_32F), mK, 
								cv::Size(mnImWidth, mnImHeight), CV_32F, mX, mY);
	else
		cv::initUndistortRectifyMap(mK, mD, cv::Mat::eye(3,3,CV_32F), mK, 
								cv::Size(mnImWidth, mnImHeight), CV_32F, mX, mY);

	invScale = 1.0/sqrt(mnImHeight* mnImHeight + mnImWidth* mnImWidth);
	//加载模型
	model_backbone = torch::jit::load(dataPath+"/Backbone.pt");
	model_descriptor = torch::jit::load(dataPath+"/Descriptor.pt");
	model_junction = torch::jit::load(dataPath+"/PointHeatmap.pt");
	model_heatmap = torch::jit::load(dataPath+"/EdgeHeatmap.pt");

    // 把模型放在gpu上
	model_backbone.to(dev);
	model_descriptor.to(dev);
	model_junction.to(dev);
	model_heatmap.to(dev);

    // 固定
	model_backbone.eval();
	model_descriptor.eval();
	model_junction.eval();
	model_heatmap.eval();

	// model_1 = torch::jit::load(dataPath+"/SOLD2Net_backbone.pt");
	// model_2 = torch::jit::load(dataPath+"/SOLD2Net_heatmap.pt");
	// model_1.to(dev);
	// model_2.to(dev);
	// model_1.eval();
	// model_2.eval();

	// super nms
	nmsFlag = (unsigned char*)std::malloc(mnImWidth * mnImHeight);
	// warm up GPU
    cv::Mat image = cv::Mat::ones(mnImHeight,mnImWidth,CV_8UC1);
	input_tensor = torch::from_blob(image.data, {1, 1, image.rows, image.cols}, torch::kByte).to(dev).toType(torch::kFloat32)/ 225.0;
	featureMap = model_backbone.forward({input_tensor}).toTensor();
	junctions = model_junction.forward({featureMap}).toTensor();
	descriptors = model_descriptor.forward({featureMap}).toTensor();
	heatmap = model_heatmap.forward({featureMap}).toTensor();

	// featureMap2 = model_1.forward({input_tensor}).toTensor();
	// torch::Tensor ts2 = model_2.forward({featureMap2}).toTensor();
}

PPGExtractor::~PPGExtractor()
{
}

void PPGExtractor::run(cv::Mat srcMat, std::vector<KeyPointEx>& _keypoints, std::vector<KeyPointEx>& _keypoints_un, std::vector<KeyEdge>& _keyedges, cv::Mat &_descriptors)
{
	assert(srcMat.channels()==1); //只支持单通道数据
	inference(srcMat);
	torch::cuda::synchronize();
	detectKeyPoint();
	// std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
	detectLines();
	// optimizeJunctions();
	// std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
	// double ttrack= std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();
	// std::cout << "detect lines time: " << ttrack << std::endl;
	genPointDescriptor();

	_keypoints = mvKeyPoints;
	_keyedges = mvKeyEdges;
	unsigned int pointSize = mvKeyPoints.size();
	_descriptors = cv::Mat(pointSize, DESC_DIM_SIZE, CV_32FC1);
	for(unsigned int i=0; i<pointSize; i++)
	{
		float *ptr_p = _descriptors.ptr<float>(i);
		void* ptr_desc = normDesc[i].data_ptr();
		std::memcpy((void*)ptr_p, ptr_desc, DESC_DIM_SIZE*sizeof(float));
	}

	if(!mbFisheye)
	{
		for(unsigned int jid=0; jid< pointSize; jid++)
			_keypoints[jid].mPos = _keypoints[jid].mPosUn;
	}
	_keypoints_un = _keypoints;
}

void PPGExtractor::inference(cv::Mat src)
{
	input_tensor = torch::from_blob(src.data, {1, 1, src.rows, src.cols}, torch::kByte).to(dev).toType(torch::kFloat32)/ 255.0;
	featureMap = model_backbone.forward({input_tensor}).toTensor();
	junctions = model_junction.forward({featureMap}).toTensor();
	heatmap = model_heatmap.forward({featureMap}).toTensor();
	descriptors = model_descriptor.forward({featureMap}).toTensor();

	// featureMap2 = model_1.forward({input_tensor}).toTensor();
	// heatmap = model_2.forward({featureMap2}).toTensor();
}

void PPGExtractor::detectKeyPoint()
{
    // non maximum suppression #cpu#
	torch::Tensor junc_norm = torch::softmax(junctions, 1);
    junc_pred = torch::pixel_shuffle(junc_norm.narrow(1,0,64),8)[0][0].cpu();
	assert(mnImHeight == junc_pred.size(0) && mnImWidth == junc_pred.size(1));

	tensorAccer2d j_ptr = junc_pred.accessor<float,2>();

	std::vector<std::tuple<unsigned int, unsigned int, float>> pointWithScore;
	pointWithScore.reserve(0.2*mnImHeight*mnImWidth);
	for(int i=0;i< mnImHeight; i++)
	{
		for(int j=0;j< mnImWidth; j++)
		{
			float thisScore = j_ptr[i][j];
			if(thisScore < JUNCTION_THRESH)
				continue;
			pointWithScore.emplace_back(j,i,thisScore);
		}
	}
	// super nms

	std::sort(pointWithScore.begin(),pointWithScore.end(), [](auto a, auto b){return std::get<2>(a) > std::get<2>(b);});
	std::memset(nmsFlag,0, mnImHeight*mnImWidth);

	mvKeyPoints.clear();
	mvKeyPoints.reserve(JUNCTION_MAX_NUM);
	for(auto kp : pointWithScore)
	{
		int posx = std::get<0>(kp);
		int posy = std::get<1>(kp);
		if( posx < JUNCTION_NMS_RADIUS || posx >(mnImWidth-JUNCTION_NMS_RADIUS-1) ||
			posy < JUNCTION_NMS_RADIUS || posy >(mnImHeight-JUNCTION_NMS_RADIUS-1) ||
			nmsFlag[posy * mnImWidth + posx] != 0)
			continue;
		nmsFlag[posy * mnImWidth + posx] = 1;
		mvKeyPoints.emplace_back(posx, posy, std::get<2>(kp));
		if(mvKeyPoints.size()+1 > JUNCTION_MAX_NUM)
			break;
		for(int i=posy-JUNCTION_NMS_RADIUS; i<=posy+JUNCTION_NMS_RADIUS; i++)
		{
			for(int j=posx-JUNCTION_NMS_RADIUS; j<=posx+JUNCTION_NMS_RADIUS;j++)
			{
				if(i<0 || i>mnImHeight || j<0 || j>mnImWidth)
					continue;
				nmsFlag[i * mnImWidth + j] = -1;
			}
		}
	}

	// undistort keypoints
	const unsigned int pointSize = mvKeyPoints.size();
	if(pointSize==0)
		return;
	cv::Mat mat(pointSize,2,CV_32F);
	for(unsigned int i=0; i<pointSize; i++)
	{
		mat.at<float>(i,0)=mvKeyPoints[i].mPos[0];
		mat.at<float>(i,1)=mvKeyPoints[i].mPos[1];
	}
	// Undistort points
	mat=mat.reshape(2);
	if(mbFisheye)
		cv::fisheye::undistortPoints(mat, mat, mK, mD, cv::Mat(), mK);
	else
		cv::undistortPoints(mat, mat, mK, mD, cv::Mat(), mK);
	mat=mat.reshape(1);
	// Fill undistorted keypoint vector
	for(unsigned int i=0; i<pointSize; i++)
	{
		float u = mat.at<float>(i,0);
		float v = mat.at<float>(i,1);
		if(u>=1 && u < mnImWidth-1 && v>=1 && v<mnImHeight-1) // 1 is the border of edge mat
			mvKeyPoints[i].mbOut = false;
		else
			mvKeyPoints[i].mbOut = true;
		mvKeyPoints[i].mPosUn << mat.at<float>(i,0),  mat.at<float>(i,1);
	}
}

void PPGExtractor::detectLines()
{
	const unsigned int pointSize = mvKeyPoints.size();
	if(pointSize==0)
		return;
	// refine heatmap
	heatmap_score = torch::softmax(heatmap, 1).select(1,1)[0].cpu();
	int gridsz_x = mnImWidth / HEATMAP_REFINE_SZ;
	int gridsz_y = mnImHeight / HEATMAP_REFINE_SZ;
	for(int i=0; i<gridsz_y; i++)
	{
		for(int j=0; j<gridsz_x; j++)
		{
			int	h_start = i*HEATMAP_REFINE_SZ;
            int w_start = j*HEATMAP_REFINE_SZ;
            int h_end = i==(gridsz_y-1) ? mnImHeight : (i+1)*HEATMAP_REFINE_SZ;
            int w_end = j==(gridsz_x-1) ? mnImWidth : (j+1)*HEATMAP_REFINE_SZ;
			torch::Tensor part_of_heatmap = heatmap_score.slice(0,h_start,h_end).slice(1,w_start,w_end);
			refineHeatMap(part_of_heatmap);
		}
	}
	// showTensor(heatmap_score);
	// undistort heatmap
	heatMat = cv::Mat(heatmap_score.size(0),heatmap_score.size(1),CV_32F);
	std::memcpy((void*)heatMat.data,heatmap_score.data_ptr(),heatmap_score.numel()*sizeof(float));
	if(mD.at<float>(0)!=0.0)
		remap(heatMat, heatMat, mX, mY, cv::INTER_LINEAR);
	cv::cv2eigen(heatMat, eigenHeat);

	Eigen::MatrixXf distMat,dirMat;
	distMat.resize(pointSize,pointSize);
	dirMat.resize(pointSize,pointSize);
	for(unsigned int i=0;i< pointSize;i++)
	{
		if(mvKeyPoints[i].mbOut)
			continue;
		Eigen::Vector2f &Ji = mvKeyPoints[i].mPosUn;
		for(unsigned int j=i+1;j<pointSize;j++)
		{
			if(mvKeyPoints[j].mbOut)
				continue;
			Eigen::Vector2f &Jj =  mvKeyPoints[j].mPosUn;
			Eigen::Vector2f vecCur = Jj-Ji;
			float dist = vecCur.norm();
			vecCur = vecCur / dist;
			distMat(i,j) = dist;
			distMat(j,i) = dist;
			dirMat(i,j) = std::atan2(vecCur.y(),vecCur.x());
			dirMat(j,i) = dirMat(i,j) - CV_PI;
			if(dirMat(j,i) < -CV_PI)
				dirMat(j,i) +=CV_2PI;
		}
	}
	// get candidate lines
	std::vector<KeyEdge> candidateLines;
	candidateLines.reserve(pointSize*pointSize*LINE_VALID_RATIO);
	std::vector<std::vector<unsigned int>> mvConnected(pointSize, std::vector<unsigned int>());
	for(unsigned int i=0;i< pointSize;i++)
	{
		KeyPointEx &kpi = mvKeyPoints[i];
		if(kpi.mbOut) 
			continue;
		for(unsigned int j=i+1;j<pointSize;j++)
		{
			KeyPointEx &kpj = mvKeyPoints[j];
			if(kpj.mbOut) 
				continue;
			// check inlier
			Eigen::Vector2f checkPoint1 = kpj.mPosUn*0.2 + kpi.mPosUn*0.8;
			Eigen::Vector2f checkPoint2 = kpj.mPosUn*0.8 + kpi.mPosUn*0.2;
			Eigen::Vector2f checkPoint3 = kpj.mPosUn*0.5 + kpi.mPosUn*0.5;
			
			if(eigenHeat((int)(checkPoint1[1]+0.5) , (int)(checkPoint1[0]+0.5)) < LINE_HEATMAP_THRESH)
				continue;	
			if(eigenHeat((int)(checkPoint2[1]+0.5) , (int)(checkPoint2[0]+0.5)) < LINE_HEATMAP_THRESH)
				continue;
			if(eigenHeat((int)(checkPoint3[1]+0.5) , (int)(checkPoint3[0]+0.5)) < LINE_HEATMAP_THRESH)
				continue;
			// check overlap
			bool isOverlap(false);
			for(unsigned int lid : mvConnected[i])
			{
				KeyEdge &line_old = candidateLines[lid];
				if(line_old.isBad)
					continue;
				unsigned int pid_old = line_old.theOtherPid(i);
				float angleDiff = dirMat(i,j) - dirMat(i,pid_old);
				if(angleDiff <-CV_PI) angleDiff+=CV_2PI;
				if(angleDiff > CV_PI) angleDiff-=CV_2PI;
				angleDiff = abs(angleDiff);
				if(angleDiff > 0.2*CV_PI)
					continue;
				float distNew = distMat(i,j);
				float distOld = distMat(i,pid_old);
				float sinAngleDiff = sin(angleDiff);
				if(distNew <= distOld && distNew*sinAngleDiff < LINE_DISTTHRESH)
					line_old.isBad = true;
				if(distOld < distNew && distOld*sinAngleDiff < LINE_DISTTHRESH)
					isOverlap = true;
			}
			if(isOverlap)
				continue;
			for(unsigned int lid : mvConnected[j])
			{
				KeyEdge &line_old = candidateLines[lid];
				if(line_old.isBad)
					continue;
				unsigned int pid_old = line_old.theOtherPid(j);
				float angleDiff = dirMat(j,i) - dirMat(j,pid_old);
				if(angleDiff <-CV_PI) angleDiff+=CV_2PI;
				if(angleDiff > CV_PI) angleDiff-=CV_2PI;
				angleDiff = abs(angleDiff);
				if(angleDiff > 0.2*CV_PI)
					continue;
				float distNew = distMat(j,i);
				float distOld = distMat(j,pid_old);
				float sinAngleDiff = sin(angleDiff);
				if(distNew <= distOld && distNew*sinAngleDiff < LINE_DISTTHRESH)
					line_old.isBad = true;
				if(distOld < distNew && distOld*sinAngleDiff < LINE_DISTTHRESH)
					isOverlap = true;
			}
			if(isOverlap)
				continue;
			candidateLines.emplace_back(i, j);
			unsigned int lid = candidateLines.size()-1;
			mvConnected[i].push_back(lid);
			mvConnected[j].push_back(lid);
		}
	}
	std::fill(mvConnected.begin(),mvConnected.end(), std::vector<unsigned int>());
	for(unsigned int i=0;i<candidateLines.size();i++)
	{
		KeyEdge &kl = candidateLines[i];
		if(kl.isBad)
			continue;
		const Eigen::Vector2f &ps = mvKeyPoints[kl.startIdx].mPosUn;
		const Eigen::Vector2f &pe = mvKeyPoints[kl.endIdx].mPosUn;
		float scoreInlier = heatMapInlierRate(ps, pe);
		if(scoreInlier < LINE_INLIER_RATE)
		{
			kl.isBad = true;
			continue;
		}
		float scoreHeatmap = heatMapLineScore(ps, pe);
		if(scoreHeatmap < LINE_HEATMAP_THRESH)
		{
			kl.isBad = true;
			continue;
		}
		kl.lscore = scoreInlier * scoreHeatmap;
		mvConnected[kl.startIdx].push_back(i);
		mvConnected[kl.endIdx].push_back(i);
	}
	// // 排序并保留最优6条线
	// const unsigned int MAX_LINE_COUNT = 4;
	// for(std::vector<unsigned int> &thisIndices : mvConnected)
	// {
	// 	if(thisIndices.empty() || thisIndices.size()<=MAX_LINE_COUNT)
	// 		continue;
	// 	std::sort(thisIndices.begin(), thisIndices.end(), 
	// 		[&](unsigned int i, unsigned int j){return candidateLines[i].lscore > candidateLines[j].lscore;});
	// 	while (thisIndices.size() > MAX_LINE_COUNT)
	// 	{
	// 		candidateLines[thisIndices.back()].isBad = true;
	// 		thisIndices.pop_back();
	// 	}
	// }
	// 搜索共线关系
	for(unsigned int p_id=0; p_id<mvConnected.size(); p_id++)
	{
		std::vector<unsigned int> thisIndices = mvConnected[p_id];
		while(thisIndices.size()>1)
		{
			double minParallelDiff(1e9);
			int bestp1_id(-1), bestp2_id(-1), bestId(-1);
			KeyEdge &kl1 = candidateLines[thisIndices.back()];
			if(kl1.isBad)
			{
				thisIndices.pop_back();
				continue;
			}
			for(unsigned int i=0;i<thisIndices.size()-1;i++)
			{
				KeyEdge &kl2 = candidateLines[thisIndices[i]];
				if(kl2.isBad)
					continue;
				unsigned int p1_id = kl1.theOtherPid(p_id);
				unsigned int p2_id = kl2.theOtherPid(p_id);
				float angleDiff = dirMat(p_id, p1_id) - dirMat(p_id, p2_id);
				double parallelDiff = 0.5 * (distMat(p_id, p1_id) + distMat(p_id, p2_id)) * std::abs(std::sin(angleDiff));
				if(minParallelDiff > parallelDiff)
				{
					minParallelDiff = parallelDiff;
					bestId = i;
					bestp1_id = p1_id;
					bestp2_id = p2_id;
				}
			}
			if(minParallelDiff > LINE_DISTTHRESH)
			{
				thisIndices.pop_back();
				continue;
			}
			mvKeyPoints[p_id].mvColine.emplace_back(bestp1_id, bestp2_id);
			thisIndices.pop_back();
			thisIndices[bestId] = thisIndices.back();
			thisIndices.pop_back();
		}
	}
	mvKeyEdges.clear();
	for(KeyEdge &cl : candidateLines)
	{
		if(cl.isBad)
			continue;
		mvKeyEdges.push_back(cl);
		mvKeyPoints[cl.startIdx].mvConnected.push_back(mvKeyEdges.size()-1);
		mvKeyPoints[cl.endIdx].mvConnected.push_back(mvKeyEdges.size()-1);
	}
}

float PPGExtractor::connectedLineScore(unsigned int pid, float biax, float biay)
{
	KeyPointEx &kp1 = mvKeyPoints[pid];
	if(kp1.mvConnected.empty())
		return 0;
	float sumScore(0.);
	for(unsigned int lid : kp1.mvConnected)
	{
		KeyEdge &kl = mvKeyEdges[lid];
		KeyPointEx &kp2 = mvKeyPoints[kl.theOtherPid(pid)];
		Eigen::Vector2f ps = kp1.mPosUn + Eigen::Vector2f(biax,biay);
		Eigen::Vector2f pe = kp2.mPosUn;
		sumScore += heatMapLineScore(ps, pe);
	}
	return sumScore / (float)(kp1.mvConnected.size());
}

float PPGExtractor::heatMapInlierRate(const Eigen::Vector2f &ps, const Eigen::Vector2f &pe)
{
	// int lenLevel = (ps-pe).norm()* invScale *4.0;
	// int segNum = (ps-pe).norm() * invSampleGapTable[lenLevel];
	// if(segNum < 2)
	// 	segNum = 2;
	// float sampleStep = 1.0/(float)segNum;
	// int inlierCount(0);
	// for(int i=1;i<segNum;i++)
	// {
	// 	Eigen::Vector2f samplePoint = ps * sampleStep * i + pe * sampleStep * (segNum -i);
	// 	float maxScore(LINE_HEATMAP_THRESH);
	// 	for(int j=0;j<circleTableL[lenLevel];j++)
	// 	{
	// 		int posx = (int)(samplePoint.x()+0.5) +circleTableX[lenLevel][j]; // round
	// 		int posy = (int)(samplePoint.y()+0.5) +circleTableY[lenLevel][j]; // round
	// 		if(eigenHeat(posy, posx)>maxScore)
	// 		{
	// 			inlierCount++;
	// 			break;
	// 		}
	// 	}
	// }
	float dist = (ps-pe).norm();
	int lenLevel = dist* invScale *4.0;
	int segNum = dist * invSampleGapTable[lenLevel];
	float sampleStep = 1.0/(float)segNum;
	int inlierCount(0);
	for(int i=1;i<segNum;i++)
	{
		Eigen::Vector2f samplePoint = ps * sampleStep * i + pe * sampleStep * (segNum -i);
		int posx = (int)(samplePoint.x()+0.5);
		int posy = (int)(samplePoint.y()+0.5);
		if(eigenHeat(posy, posx)>LINE_HEATMAP_THRESH)
			inlierCount++;
	}
	return (float)inlierCount/ (float)(segNum-1);
}

float PPGExtractor::heatMapLineScore(const Eigen::Vector2f &ps, const Eigen::Vector2f &pe)
{
	float dist = (ps-pe).norm();
	int lenLevel = dist* invScale *4.0;
	int segNum = dist * invSampleGapTable[lenLevel];
	float sampleStep = 1.0/(float)segNum;
	float sumScore(0.);
	for(int i=1;i<segNum;i++)
	{
		Eigen::Vector2f samplePoint = ps * sampleStep * i + pe * sampleStep * (segNum -i);
		sumScore += bilinearInterpolation(eigenHeat, samplePoint.x(),samplePoint.y());
	}
	return sumScore/(float)(segNum-1);
}

void PPGExtractor::optimizeJunctions()
{
	for(unsigned int pid=0;pid<mvKeyPoints.size();pid++)
	{
		mvKeyPoints[pid].mfScore = connectedLineScore(pid);
		int iterNum = OPTIMIZE_ITER_NUM;
		float iterStep = 2;
		while(iterNum--)
		{
			KeyPointEx &kp =mvKeyPoints[pid];
			// right
			kp.updatePreturb(iterStep,0, connectedLineScore(pid,iterStep,0));
			// left
			kp.updatePreturb(-iterStep,0, connectedLineScore(pid,-iterStep,0));
			// down
			kp.updatePreturb(0,iterStep, connectedLineScore(pid,0,iterStep));
			// up
			kp.updatePreturb(0,-iterStep, connectedLineScore(pid,0,-iterStep));
			iterStep *= OPTIMIZE_ITER_DECAY;
		}
	}
}

void PPGExtractor::genPointDescriptor()
{
	// 准备描述子采样点
	int allPointNum = mvKeyPoints.size();

	if(allPointNum < 10) // 防止未检测出关键点
	{
		normDesc = torch::zeros({allPointNum,DESC_DIM_SIZE},torch::kFloat);
		return;
	}

	torch::Tensor allPoints = torch::zeros({1,allPointNum,1,2},torch::kFloat);
	tensorAccer4d p_ptr = allPoints.accessor<float,4>();
	for(unsigned int i=0; i< mvKeyPoints.size(); i++)
	{
		p_ptr[0][i][0][0] = mvKeyPoints[i].mPos[0] / (float)mnImWidth *2. -1.;
		p_ptr[0][i][0][1] = mvKeyPoints[i].mPos[1] / (float)mnImHeight *2. -1.;
	}
	allPoints = allPoints.cuda();
	torch::Tensor sampleVal = torch::squeeze(torch::grid_sampler(descriptors,allPoints,0,0,false)).permute({1,0});
	torch::Tensor normalizedVal = torch::nn::functional::normalize(sampleVal,
								  torch::nn::functional::NormalizeFuncOptions().dim(1));
	normDesc = normalizedVal.contiguous().cpu();
}

void PPGExtractor::refineHeatMap(torch::Tensor &scoreMap)
{
	std::vector<float> heatmapVal;
	tensorAccer2d s_ptr = scoreMap.accessor<float,2>();
	int segHeight = scoreMap.size(0);
	int segWidth = scoreMap.size(1);
	for(int i=0; i<segHeight; i++)
	{
		for(int j=0; j<segWidth;j++)
		{
			if(s_ptr[i][j] > LINE_VALID_THRESH)
				heatmapVal.push_back(s_ptr[i][j]);
		}
	}
	int valCount = LINE_VALID_RATIO * heatmapVal.size();
	if(valCount<1)
		return;
	if(heatmapVal.size() >=segHeight * segWidth * 0.9 && heatmapVal[heatmapVal.size()*0.9] > 0.1)
	{
		scoreMap.fill_(0);
		return;
	}
	std::sort(heatmapVal.begin(),heatmapVal.end(),[](float a,float b){return a>b;});
	float aveVal = std::accumulate(heatmapVal.begin(), heatmapVal.begin()+valCount, 0.0)/(float)valCount;
	for(int i=0; i<segHeight; i++)
	{
		for(int j=0; j<segWidth;j++)
		{
			float curScore = s_ptr[i][j];
			if(curScore > LINE_VALID_THRESH)
			{
				float newScore = s_ptr[i][j] / aveVal;
				s_ptr[i][j] = newScore >1.0 ? 1.0:newScore;
			}
			else
				s_ptr[i][j] = 0;
		}
	}
}

float PPGExtractor::bilinearInterpolation(const Eigen::MatrixXf &M, float ptX, float ptY)
{
	int x1 = (int)ptX;
	int x2 = x1 + 1;
	int y1 = (int)ptY;
	int y2 = y1 + 1;
	float data1 = (x2 - ptX) * M(y1, x1) + (ptX - x1) * M(y1, x2);
	float data2 = (x2 - ptX) * M(y2, x1) + (ptX - x1) * M(y2, x2);
	return (y2 - ptY) * data1 + (ptY - y1) * data2;
}

void PPGExtractor::showTensor(const torch::Tensor& ts)
{
	assert(ts.dim()==2);
	assert(ts.scalar_type()== torch::kFloat);
	cv::Mat showMat(ts.size(0),ts.size(1),CV_32FC1);
	std::memcpy((void*)showMat.data,ts.data_ptr(),ts.numel()*sizeof(float));
	cv::normalize(showMat,showMat,0.,1.,cv::NORM_MINMAX);
	showMat=showMat*255;
	showMat.convertTo(showMat,CV_8UC1);
	cv::resize(showMat, showMat, cv::Size(0.5*mnImWidth, 0.5*mnImHeight), cv::INTER_LINEAR);
	cv::imshow("tensorMatrix",showMat);
	cv::waitKey(1);
}