/*
 * BinarySampleMFReader.cpp
 */

#include "rudra/io/BinarySampleMFReader.h"
#include "rudra/util/MatrixContainer.h"
#include "rudra/io/BinaryMatrixReader.h"
#include "rudra/util/Logger.h"
#include "rudra/MLPparams.h"
#include <vector>
#include <algorithm>

namespace rudra {
BinarySampleMFReader::BinarySampleMFReader(std::string sampleFileName,
					   std::string labelFileName, size_t totalSampleNum, RudraRand rr) :
		sizePerImg(MLPparams::_numInputDim), tdSummaryFile(sampleFileName), tlSummaryFile(
												  labelFileName),xSize(-1), ySize(-1), totalSampleNum(totalSampleNum), rr(rr) {
    this->initXYTabs();
    this->setLabelDim(); // ugly hack to set the label dimensions;
    this->sizePerLabel = MLPparams::_labelDim;
    std::cout<<"label dimensions:" <<MLPparams::_labelDim<<std::endl;
    
}

BinarySampleMFReader::~BinarySampleMFReader() {
}

/**
 * Read a chosen number of samples into matrix X and the corresponding labels
 * into matrix Y.
 */
void BinarySampleMFReader::readLabelledSamples(size_t numSamples,
		MatrixContainer<float> &X, MatrixContainer<float>& Y) {
	std::vector<size_t> idx(numSamples);
	for (size_t i = 0; i < numSamples; ++i) {
	    idx[i] = rr.getLong() % totalSampleNum + 1; // we have a contract that idx is between 1 and totalNumSample
	}
	std::sort(idx.begin(), idx.end());
	std::string myXFile, myYFile;
	size_t myXIdx, myYIdx;
	for (size_t i = 0; i < numSamples; ++i) {
	    lookupTab(xTab, idx[i], myXFile, myXIdx);
	    lookupTab(yTab, idx[i], myYFile, myYIdx);
	    assert(myXIdx == myYIdx);
	    readRecordFromBinMat(X.buf + i * sizePerImg, myXIdx, sizePerImg, myXFile);
	    readRecordFromBinMat(Y.buf + i * sizePerLabel, myYIdx, sizePerLabel, myYFile);
	}
}

    /**
     * initialize X Y tables
     */
    void BinarySampleMFReader:: initXYTabs(){
	this->populateTable(tdSummaryFile, xTab, xSize);
	this->populateTable(tlSummaryFile, yTab, ySize);
	if(xSize != ySize){
	    std::stringstream xSizeSS;
	    std::stringstream ySizeSS;
	    xSizeSS<<xSize;
	    ySizeSS<<ySize;
	    Logger::logFatal("number of samples in xFile and yFile are not the same! xSize:"+xSizeSS.str()+" ySize:"+ySizeSS.str()+ " [BinarySampleMFReader]\n");
	}
	if(xSize != totalSampleNum){
	    Logger::logFatal("number of samples in xFile and yFile are not equal to total sample number! [BinarySampleMFReader]\n");
	}
	this->checkTabConsistency();
    }

    void BinarySampleMFReader::setLabelDim(){
	std::map<size_t, std::string>::iterator labelTabIt = yTab.begin();
	std::string fName = labelTabIt->second;
	int rows, columns;
        MLPparams::readBinHeader(fName, rows, columns);
	MLPparams::setLabelDim(columns);
    }

    /**
     * check table consistency
     */
    void BinarySampleMFReader::checkTabConsistency(){
	// check if X,Y tables are consistent
	if(xTab.size() != yTab.size()){
	    Logger::logFatal("x y tabs size dont match [BinarySampleMFReader] \n");
	}
	std::map<size_t,std::string>::iterator xIt;
	std::map<size_t, std::string>::iterator yIt;
	for (xIt = xTab.begin(), yIt = yTab.begin(); (xIt!=xTab.end()) && (yIt!=yTab.end()); ++xIt, ++yIt){
	    if( xIt->first != yIt->first){
		Logger::logFatal(""+xIt->second+" and " + yIt->second + " size doesn't match [BinarySampleMFReader] \n");
	    }
	}
	// check if |X| = |Y| = MLPparams::_numSamples

    }

    void BinarySampleMFReader::populateTable(std::string fileName, std::map<size_t, std::string> &tab, size_t &size){
	std::ifstream f1(fileName.c_str(), std::ios::in); //open file for reading
	if (!f1) {
	    Logger::logFatal(fileName+"does not exist [BinarySampleMFReader]\n");
	}
	size_t curSize = 0;
	std::string line;
	int _num = 0;
	while (!f1.eof()) {
	    getline(f1, line, '\n');
	    _num++;
	    if(line.empty()){
		continue;
	    }
	    std::ifstream tmp(line.c_str(), std::ios::in); //open file for reading
	 
	    if(!tmp){
		std::stringstream ss;
		ss<<_num;
		Logger::logFatal(fileName+ "line" + ss.str() + " file "+ line + "doesn't exist [BinarySampleMFReader]\n");
	    }
	    int rows =  -1;
	    int columns = -1;
	    MLPparams::readBinHeader(line, rows, columns);
	    std::cout<<"file: "<<line<<" rows:" <<rows<<" columns: "<<columns<<std::endl;
	    assert(rows != -1);
	    assert(columns != -1);
	    curSize += rows; 
	    tab.insert(std::pair<size_t, std::string>(curSize, line));
	}
	size = curSize;

    }

    /**
     * lookup table, assuming idx is from 1 to totalSampleNum, fIdx is from 0 to totalSampleNum -1 
     */
    void BinarySampleMFReader::lookupTab(std::map<size_t, std::string> tab, size_t idx, std::string &fName, size_t &fIdx){
	assert((idx >=1 && idx <=totalSampleNum));
	std::map<size_t, std::string>::iterator it;
	it = tab.lower_bound (idx);  // itlow points to b
	if( it  == tab.end()){
	    Logger::logFatal("[BinarySampleMFReader] looking for unreasable idx.\n");
	}
	if(it == tab.begin()){ // in the first file
	    fName = it->second;
	    fIdx = idx - 1;
	}else{
	    fName = it->second;
	    it--;
	    size_t prevSize = it->first;
	    fIdx = idx - prevSize - 1;
	}
	assert(fIdx >= 0);
    }

    

} /* namespace rudra */
