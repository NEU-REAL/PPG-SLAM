#include "PPGGraph.h"

unsigned long int MapEdge::mnNextId = 0;
unsigned long int MapColine::mnNextId = 0;
double MapEdge::viewCosTh = 0.1;

MapEdge::MapEdge(MapPoint* ps, MapPoint* pe) : 
	mpMPs(ps), mpMPe(pe), mbBad(false), mbValid(true)
{
	mnBALocalForKF = 0;
	trackedFrameId = 0;
	mnId = mnNextId++;
	ps->addEdge(this);
	pe->addEdge(this);
	startTime = chrono::steady_clock::now();
}

MapPoint* MapEdge::theOtherPt(MapPoint* pMP)
{
	if(mpMPs == pMP)
		return mpMPe;
	if(mpMPe == pMP)
		return mpMPs;
	return nullptr;
}

void MapEdge::addObservation(KeyFrame* pKF, unsigned int keyId)
{
	std::unique_lock<std::mutex> lock(mtxObs);
	mObservations[pKF] = keyId;
}

std::map<KeyFrame*, int> MapEdge::getObservations()
{
	std::unique_lock<std::mutex> lock(mtxObs);
	return mObservations; 
}

void MapEdge::checkValid()
{
	auto obs = getObservations();
	if(obs.size() < 2)
	{
		mbValid = false;
		return;
	}
	// CHECK LINE DIRECTION
	Eigen::Vector3f n1_ = mpMPs->GetNormal().normalized();
	Eigen::Vector3f n2_ = mpMPe->GetNormal().normalized();
	Eigen::Vector3f v_ = (mpMPs->GetWorldPos() - mpMPe->GetWorldPos()).normalized();
	float cosVeiw1 = v_.dot(n1_);
	float cosVeiw2 = v_.dot(n2_);
	if(std::fabs(cosVeiw1) > MapEdge::viewCosTh || std::fabs(cosVeiw2) > MapEdge::viewCosTh)
		mbValid = false;
	else
		mbValid = true;
}

bool MapEdge::isBad()
{
	std::unique_lock<std::mutex> lock(mtxObs);
	return (mbBad || mpMPs->isBad() || mpMPe->isBad());
}

MapColine::MapColine(MapPoint* pMPs, MapPoint* pMPm, MapPoint* pMPe) : 
	mpMPs(pMPs), mpMPm(pMPm), mpMPe(pMPe), mbBad(false), mbValid(false), mpFirstKF(nullptr)
{
	mnId = mnNextId++;
}

void MapColine::addObservation(KeyFrame* pKF, float weight)
{
	std::unique_lock<std::mutex> lock(mtxObs);
	if (mObservations.count(pKF))
		return;
	if(mObservations.empty())
		mpFirstKF = pKF;
	mObservations[pKF] = weight;
	// check valid
	if(mObservations.size() < 2 || mbValid)
		return;
	Eigen::Vector3f pts = mpMPs->GetWorldPos();
	Eigen::Vector3f pte = mpMPe->GetWorldPos();
	Eigen::Vector3f posKF_ini = mpFirstKF->GetCameraCenter();
	Eigen::Vector3f posKF_cur = pKF->GetCameraCenter();
	Eigen::Vector3f n1_ = (pts-pte).cross(posKF_ini).normalized();
	Eigen::Vector3f n2_ = (pts-pte).cross(posKF_cur).normalized();
	if( std::fabs(n1_.dot(n2_)) < 1)
		mbValid = true;
}

float MapColine::aveWeight()
{
	std::unique_lock<std::mutex> lock(mtxObs);
	float ret = 0;
	for(auto mmC : mObservations)
		ret += mmC.second;
	return ret;
}

std::map<KeyFrame*, int> MapColine::getObservations()
{
	std::unique_lock<std::mutex> lock(mtxObs);
	return mObservations;
}

bool MapColine::isBad()
{
	if(mpMPs->mpReplaced)
		mpMPs = mpMPs->mpReplaced;
	if(mpMPe->mpReplaced)
		mpMPe = mpMPe->mpReplaced;
	return mbBad || mpMPs->isBad() || mpMPm->isBad() || mpMPe->isBad();
}