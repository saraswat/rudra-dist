/*
 * UnifiedBinarySampleReader.cpp
 *
 *  Created on: Jul 13, 2015
 *      Author: weiz
 */

#include "rudra/io/UnifiedBinarySampleReader.h"
#include "rudra/util/MatrixContainer.h"
#include "rudra/MLPparams.h"
#include "rudra/util/RudraRand.h"
#include "rudra/util/Logger.h"
#include <vector>
#include <algorithm>
#include <iostream>
#include <vector>
#include <string>
#include <cstdlib>
namespace rudra {
UnifiedBinarySampleReader::UnifiedBinarySampleReader(std::string sampleFileName,
				       std::string labelFileName, RudraRand rr) :
		sizePerImg(MLPparams::_numInputDim), tdFile(sampleFileName), tlFile(
										    labelFileName),rr(rr) {
    this->setLabelDim();
    this->sizePerLabel = MLPparams::_labelDim;
    this->checkFiles();
    this->initFileTypes();
}

void UnifiedBinarySampleReader::checkFiles(){
	std::ifstream fx(tdFile.c_str(), std::ios::in | std::ios::binary);
	if(!fx){
		Logger::logFatal(tdFile+" doesn't exist");
		exit(EXIT_FAILURE);
	}
	std::ifstream fy(tlFile.c_str(), std::ios::in | std::ios::binary);
	if(!fy){
		Logger::logFatal(tlFile+" doesn't exist");
		exit(EXIT_FAILURE);
	}
}

void UnifiedBinarySampleReader::initFileTypes(){
	std::string xExt = getFileExt(tdFile);
	std::string yExt = getFileExt(tlFile);
	tdFT = lookupFileType(xExt);
	tlFT = lookupFileType(yExt);
	if(INVALID == tdFT || INVALID == tlFT){
		Logger::logFatal("Wrong files extension");
		exit(EXIT_FAILURE);
	}
}
std::string UnifiedBinarySampleReader::getFileExt(const std::string& s) {

   size_t i = s.rfind('.', s.length());
   if (i != std::string::npos) {
      return(s.substr(i+1, s.length() - i));
   }

   return("");
}

BinFileType UnifiedBinarySampleReader::lookupFileType(const std::string& s){
	if(s.compare("bin") == 0){
		return FLOAT;
	}
	if(s.compare("bin8") == 0){
		return CHAR;
	}
	if(s.compare("bin32") == 0){
		return INT;
	}
	return INVALID;
}

UnifiedBinarySampleReader::~UnifiedBinarySampleReader() {
}

/**
 * Read a chosen number of samples into matrix X and the corresponding labels
 * into matrix Y.
 */
void UnifiedBinarySampleReader::readLabelledSamples(size_t numSamples,
		MatrixContainer<float> &X, MatrixContainer<float>& Y) {
	std::vector<size_t> idx(numSamples);
	for (size_t i = 0; i < numSamples; ++i) {
	    idx[i] = rr.getLong() % MLPparams::_numTrainSamples;
	}
	std::sort(idx.begin(), idx.end());

	// training data
	switch(tdFT){
		case FLOAT:{
			MatrixContainer<float> tX(MLPparams::_batchSize, MLPparams::_numInputDim);
			for(size_t i = 0; i < numSamples; ++i){
				readBinMat<float>(tX.buf + i * sizePerImg, idx[i], 1, sizePerImg, tdFile);
			}
			X = tX;
			break;
		}

		case CHAR:{
			MatrixContainer<uint8> tX(MLPparams::_batchSize, MLPparams::_numInputDim);
			for(size_t i = 0; i < numSamples; ++i){
				readBinMat<uint8>(tX.buf + i * sizePerImg, idx[i], 1, sizePerImg, tdFile);
			}
			X = tX;
			break;
		}

		case INT:{
			//TODO
			Logger::logFatal("Training data file type of INT is not supported yet");
			exit(EXIT_FAILURE);
			break;
		}
		default :{
			Logger::logFatal("Training data file type is invalid!");
			exit(EXIT_FAILURE);
			break;
		}

	}
	// training label

	switch(tlFT){
		case FLOAT:{
			MatrixContainer<float> tY(MLPparams::_batchSize, MLPparams::_labelDim);
			for(size_t i = 0; i < numSamples; ++i){
				readBinMat<float>(tY.buf + i * sizePerLabel, idx[i], 1, sizePerLabel, tlFile);
			}
			Y = tY;
			break;
		}

		case CHAR:{
			MatrixContainer<uint8> tY(MLPparams::_batchSize, MLPparams::_labelDim);
			for(size_t i = 0; i < numSamples; ++i){
				readBinMat<uint8>(tY.buf + i * sizePerLabel, idx[i], 1, sizePerLabel, tlFile);
			}
			Y = tY;
			break;
		}

		case INT:{
			//TODO
			Logger::logFatal("Training label file type of INT is not supported yet");
			exit(EXIT_FAILURE);
			break;
		}
		default :{
			Logger::logFatal("Training label file type is invalid!");
			exit(EXIT_FAILURE);
			break;
		}

	}

}


    void UnifiedBinarySampleReader::setLabelDim(){
    	int rows, columns;
        MLPparams::readBinHeader(tlFile, rows, columns);
        MLPparams::setLabelDim(columns);
    }

}
