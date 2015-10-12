/*
 * UnifiedBinarySampleReader.cpp
 *
 *  Created on: Jul 13, 2015
 *      Author: weiz
 */

#include "rudra/io/UnifiedBinarySampleReader.h"
#include "rudra/io/BinaryMatrixReader.h"
#include "rudra/MLPparams.h"
#include "rudra/util/RudraRand.h"
#include "rudra/util/Logger.h"
#include <algorithm>
#include <iostream>
#include <cstdlib>

namespace rudra {
UnifiedBinarySampleReader::UnifiedBinarySampleReader(std::string sampleFileName,
		std::string labelFileName, RudraRand rr) :
		SampleReader(MLPparams::_numInputDim), tdFile(sampleFileName), tlFile(
				labelFileName), rr(rr) {
    this->initSizePerLabel();
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
void UnifiedBinarySampleReader::readLabelledSamples(const size_t numSamples, float* X,
		float* Y) {
	std::vector<size_t> idx(numSamples);
	for (size_t i = 0; i < numSamples; ++i) {
	    idx[i] = rr.getLong() % MLPparams::_numTrainSamples;
	}
	std::sort(idx.begin(), idx.end());

	retrieveData(numSamples, idx, X, Y);
}

void UnifiedBinarySampleReader::retrieveData(const size_t numSamples,
		const std::vector<size_t>& idx, float* X, float* Y) {
	switch (tdFT) {
	case FLOAT: {
		readRecordsFromBinMat(X, numSamples, idx, sizePerSample, tdFile);
		break;
	}

	case CHAR: {
		uint8* tempX = new uint8[numSamples * sizePerSample];
		readRecordsFromBinMat(tempX, numSamples, idx, sizePerSample, tdFile);
		for (size_t i = 0; i < numSamples * sizePerSample; ++i) {
			X[i] = tempX[i]; // convert from uint8 to float
		}
		delete[] tempX;
		break;
	}

	case INT: {
		//TODO
		Logger::logFatal("Training data file type of INT is not supported yet");
		exit(EXIT_FAILURE);
		break;
	}
	default: {
		Logger::logFatal("Training data file type is invalid!");
		exit(EXIT_FAILURE);
		break;
	}

	}
	// training label
	switch (tlFT) {
	case FLOAT: {
		readRecordsFromBinMat(Y, numSamples, idx, sizePerLabel, tlFile);
		break;
	}

	case CHAR: {
		uint8* tempY = new uint8[numSamples * sizePerLabel];
		readRecordsFromBinMat(tempY, numSamples, idx, sizePerLabel, tlFile);
		for (size_t i = 0; i < numSamples * sizePerLabel; ++i) {
			Y[i] = tempY[i]; // convert from uint8 to float
		}
		delete[] tempY;
		break;
	}

	case INT: {
		//TODO
		Logger::logFatal(
				"Training label file type of INT is not supported yet");
		exit(EXIT_FAILURE);
		break;
	}
	default: {
		Logger::logFatal("Training label file type is invalid!");
		exit(EXIT_FAILURE);
		break;
	}

	}
}

void UnifiedBinarySampleReader::initSizePerLabel() {
	int rows, columns;
	MLPparams::readBinHeader(tlFile, rows, columns);
	sizePerLabel = columns;
}

}
