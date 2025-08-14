#include <ctime>
#include <iostream>
#include <unistd.h>
#include <fstream>
#include <chrono>
#include <opencv2/opencv.hpp>
#include "feature/include/PPGExtractor.h"
#include "sensors/include/GeometricCamera.h"
#include "sensors/include/Pinhole.h"
#include "DBoW3/DBoW3.h"

std::vector<cv::Mat> getFeatures(PPGExtractor *pExt, std::string dataPath, int n = 1e6)
{
	std::vector<cv::Mat> features;
	std::ifstream imgFile(dataPath + "/mav0/cam0/data.csv", std::ifstream::in);
	std::string str_line, str_ts, str_path;
	getline(imgFile, str_line);
	cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(8, 8));
	while (!imgFile.eof())
	{
		getline(imgFile, str_line);
		if(str_line.empty())
			break;
		std::size_t pos = str_line.find(",", 0);
		str_ts = str_line.substr(0, pos);
		std::string imgPath = dataPath + "/mav0/cam0/data/" + str_ts + ".png";
		cv::Mat image = cv::imread(imgPath, 0);
		if (image.empty() || n-- < 0)
			break;
		if(n%4 != 0)
			continue;
		clahe->apply(image, image);
		cv::Mat mPointDesc;
		std::vector<KeyPointEx> keyPoints, keyPointsUn;
		std::vector<KeyEdge> keyEdges;
		pExt->run(image, keyPoints,keyPointsUn, keyEdges, mPointDesc);
		features.push_back(mPointDesc.clone());
		std::cerr<<imgPath << ":"<<keyPoints.size()<<std::endl;
		cv::Mat showMat;
		cv::cvtColor(image,showMat, cv::COLOR_GRAY2BGR);
		for(KeyPointEx kp : keyPoints)
			cv::circle(showMat, cv::Point(kp.mPos[0],kp.mPos[1]),2, cv::Scalar(0,255,0), -1);
		cv::imshow("aaa", showMat);
		cv::waitKey(10);
	}
	std::cerr << dataPath << " size " << features.size() << std::endl;
	return features;
}

int main()
{
	cv::FileStorage fsSettings("config/EuRoC.yaml", cv::FileStorage::READ);
    int width = fsSettings["Camera.width"].operator int();
    int height = fsSettings["Camera.height"].operator int();
    float fps = fsSettings["Camera.fps"].real();
    GeometricCamera* pCam;
    std::vector<float> vCalibration(8,0);
	vCalibration[0] = fsSettings["Camera.fx"].real();
	vCalibration[1] = fsSettings["Camera.fy"].real();
	vCalibration[2] = fsSettings["Camera.cx"].real();
	vCalibration[3] = fsSettings["Camera.cy"].real();
	vCalibration[4] = fsSettings["Camera.k1"].real();
	vCalibration[5] = fsSettings["Camera.k2"].real();
	vCalibration[6] = fsSettings["Camera.p1"].real();
	vCalibration[7] = fsSettings["Camera.p2"].real();
	pCam = new Pinhole(vCalibration, width, height, fps);
	PPGExtractor ext(pCam, "net");

	// std::vector<cv::Mat> featureMag1 = getFeatures(&ext, "/home/z/Datasets/euroc/MH_01_easy");
	// std::vector<cv::Mat> featureMag2 = getFeatures(&ext, "/home/z/Datasets/euroc/MH_02_easy");
	// std::vector<cv::Mat> featureMag3 = getFeatures(&ext, "/home/z/Datasets/euroc/MH_03_medium");
	std::vector<cv::Mat> featureMag4 = getFeatures(&ext, "/home/z/Datasets/euroc/MH_04_difficult");
	// std::vector<cv::Mat> featureMag5 = getFeatures(&ext, "/home/z/Datasets/euroc/MH_05_difficult");

	// std::vector<cv::Mat> featureMag1 = getFeatures(&ext, "/home/z/Datasets/tum/dataset-magistrale1_512_16");
	// std::vector<cv::Mat> featureMag2 = getFeatures(&ext, "/home/z/Datasets/tum/dataset-magistrale2_512_16");
	// std::vector<cv::Mat> featureMag3 = getFeatures(&ext, "/home/z/Datasets/tum/dataset-magistrale3_512_16");
	// std::vector<cv::Mat> featureMag4 = getFeatures(&ext, "/home/z/Datasets/tum/dataset-magistrale4_512_16");
	// std::vector<cv::Mat> featureMag5 = getFeatures(&ext, "/home/z/Datasets/tum/dataset-magistrale5_512_16");
	// std::vector<cv::Mat> featureMag6 = getFeatures(&ext, "/home/z/Datasets/tum/dataset-magistrale5_512_16");
	std::vector<cv::Mat> allfeatures;
	allfeatures.insert(allfeatures.end(), featureMag4.begin(), featureMag4.end());
	// allfeatures.insert(allfeatures.end(), featureMag5.begin(), featureMag5.end());
	// allfeatures.insert(allfeatures.end(), featureMag3.begin(), featureMag3.end());
	// allfeatures.insert(allfeatures.end(), featureMag4.begin(), featureMag4.end());
	// allfeatures.insert(allfeatures.end(), featureMag5.begin(), featureMag5.end());
	// allfeatures.insert(allfeatures.end(), featureMag6.begin(), featureMag6.end());
	{
		DBoW3::Vocabulary voc1(20, 3, DBoW3::TF_IDF, DBoW3::L2_NORM);
		std::cerr << "start" << std::endl;
		std::chrono::time_point<std::chrono::system_clock> t1 = std::chrono::system_clock::now();
		voc1.create(allfeatures);
		std::chrono::time_point<std::chrono::system_clock> t2 = std::chrono::system_clock::now();
		unsigned long int dur1 = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
		std::cerr << "end 10x1 " << dur1 * 0.001 << std::endl;
		voc1.save("voc_tum_10x1.gz");
	}
	return 0;
}

// int main()
// {
// 	Settings setting_("TUM-VI.yaml");
// 	cv::Size imSzz = setting_.imSize();
// 	PPGExtractor ext(imSzz.width, imSzz.height, setting_.camera()->toK(), cv::Mat::zeros(4, 1, CV_32F), "net", false);
 
// 	std::vector<cv::Mat> featureMH1 = getFeatures(&ext, "/home/z/Datasets/tum/dataset-magistrale1_512_16", 2000);
// 	{
// 		DBoW3::Vocabulary voc1("Vocabulary/voc_tum_9x3.gz");
// 		DBoW3::Database DB(voc1, false);
// 		for(unsigned int i=0;i<featureMH1.size();i++)
// 			DB.add(featureMH1[i]);
// 		DBoW3::QueryResults rst;
// 		DB.query(featureMH1[450], rst, 20);
// 		for(unsigned int i=0;i<rst.size();i++)
// 		{
// 			std::cout<<i<<" "<<rst[i].Id<<" "<<rst[i].Score<<std::endl;
// 			cv::Mat img = featureMH1[rst[i].Id];
// 			cv::imshow("img" + std::to_string(rst[i].Id), img);
// 			cv::waitKey(0);
// 		}
// 	}
// 	return 0;
// }

// int main()
// {
// 	CameraGeneric *camera = new Pinhole(0,"settings_euroc.yaml");
// 	SOLD2net net(camera, "sold2_net");

// 	std::vector<cv::Mat> featureMH1 = getFeatures(&net,camera,"/home/z/Datasets/MH_01_easy");
// 	std::vector<cv::Mat> featureMH2 = getFeatures(&net,camera,"/home/z/Datasets/MH_02_easy");
// 	std::vector<cv::Mat> featureMH3 = getFeatures(&net,camera,"/home/z/Datasets/MH_03_medium");
// 	std::vector<cv::Mat> featureMH4 = getFeatures(&net,camera,"/home/z/Datasets/MH_04_difficult");
// 	std::vector<cv::Mat> featureMH5 = getFeatures(&net,camera,"/home/z/Datasets/MH_05_difficult");

// 	std::vector<cv::Mat> featureV11 = getFeatures(&net,camera,"/home/z/Datasets/V1_01_easy");
// 	std::vector<cv::Mat> featureV12 = getFeatures(&net,camera,"/home/z/Datasets/V1_02_medium");
// 	std::vector<cv::Mat> featureV13 = getFeatures(&net,camera,"/home/z/Datasets/V1_03_difficult");
// 	std::vector<cv::Mat> featureV21 = getFeatures(&net,camera,"/home/z/Datasets/V2_01_easy");
// 	std::vector<cv::Mat> featureV22 = getFeatures(&net,camera,"/home/z/Datasets/V2_02_medium");
// 	std::vector<cv::Mat> featureV23 = getFeatures(&net,camera,"/home/z/Datasets/V2_03_difficult");

// 	std::vector<cv::Mat> allfeatures;
// 	allfeatures.insert(allfeatures.end(), featureMH1.begin(), featureMH1.end());
// 	allfeatures.insert(allfeatures.end(), featureMH2.begin(), featureMH2.end());
// 	allfeatures.insert(allfeatures.end(), featureMH3.begin(), featureMH3.end());
// 	allfeatures.insert(allfeatures.end(), featureMH4.begin(), featureMH4.end());
// 	allfeatures.insert(allfeatures.end(), featureMH5.begin(), featureMH5.end());
// 	allfeatures.insert(allfeatures.end(), featureV11.begin(), featureV11.end());
// 	allfeatures.insert(allfeatures.end(), featureV12.begin(), featureV12.end());
// 	allfeatures.insert(allfeatures.end(), featureV13.begin(), featureV13.end());
// 	allfeatures.insert(allfeatures.end(), featureV21.begin(), featureV21.end());
// 	allfeatures.insert(allfeatures.end(), featureV22.begin(), featureV22.end());
// 	allfeatures.insert(allfeatures.end(), featureV23.begin(), featureV23.end());

// 	{
// 		DBoW3::Vocabulary voc1(9, 3, DBoW3::TF_IDF, DBoW3::L2_NORM);
// 		std::cerr<<"start"<<std::endl;
// 		std::chrono::time_point<std::chrono::system_clock> t1 = std::chrono::system_clock::now();
// 		voc1.create(allfeatures);
// 		std::chrono::time_point<std::chrono::system_clock> t2 = std::chrono::system_clock::now();
// 		unsigned long int dur1 = std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count();
// 		std::cerr<<"end 9x3 "<<dur1 * 0.001<<std::endl;
// 		voc1.save("voc_euroc_9x3.gz");
// 	}

// 	{
// 		DBoW3::Vocabulary voc2(20, 3, DBoW3::TF_IDF, DBoW3::L2_NORM);
// 		std::cerr<<"start"<<std::endl;
// 		std::chrono::time_point<std::chrono::system_clock> t3 = std::chrono::system_clock::now();
// 		voc2.create(allfeatures);
// 		std::chrono::time_point<std::chrono::system_clock> t4 = std::chrono::system_clock::now();
// 		unsigned long int dur2 = std::chrono::duration_cast<std::chrono::milliseconds>(t4-t3).count();
// 		std::cerr<<"end 20x3 "<<dur2 * 0.001<<std::endl;
// 		voc2.save("voc_euroc_20x3.gz");
// 	}
// 	{
// 		DBoW3::Vocabulary voc3(10, 5, DBoW3::TF_IDF, DBoW3::L2_NORM);
// 		std::cerr<<"start"<<std::endl;
// 		std::chrono::time_point<std::chrono::system_clock> t5 = std::chrono::system_clock::now();
// 		voc3.create(allfeatures);
// 		std::chrono::time_point<std::chrono::system_clock> t6 = std::chrono::system_clock::now();
// 		unsigned long int dur3 = std::chrono::duration_cast<std::chrono::milliseconds>(t6-t5).count();
// 		std::cerr<<"end 10x5 "<<dur3 * 0.001<<std::endl;
// 		voc3.save("voc_euroc_10x5.gz");
// 	}
// 	return 0;
// }
