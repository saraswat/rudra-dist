/*
 * UnifiedBinarySampleReader.cpp
 */

#include "rudra/io/UnifiedBinarySampleReader.h"
#include "rudra/io/BinaryMatrixReader.h"
#include "rudra/util/RudraRand.h"
#include "rudra/util/Logger.h"
#include <algorithm>
#include <iostream>
#include <cstdlib>

namespace rudra {
UnifiedBinarySampleReader::UnifiedBinarySampleReader(std::string sampleFileName,
		std::string labelFileName, RudraRand rr) :
		tdFile(sampleFileName), tlFile(labelFileName), rr(rr) {
	this->checkFiles();

	SampleReader::readHeader(tdFile, numSamples, sizePerSample);
	size_t dummy;
	SampleReader::readHeader(tlFile, dummy, sizePerLabel);

	std::string xExt = getFileExt(tdFile);
	tdFT = lookupFileType(xExt);
	std::string yExt = getFileExt(tlFile);
	tlFT = lookupFileType(yExt);
}

UnifiedBinarySampleReader::~UnifiedBinarySampleReader() {
}

void UnifiedBinarySampleReader::checkFiles() {
	std::ifstream fx(tdFile.c_str(), std::ios::in | std::ios::binary);
	if (!fx) {
		Logger::logFatal(tdFile + " doesn't exist");
		exit(EXIT_FAILURE);
	}
	std::ifstream fy(tlFile.c_str(), std::ios::in | std::ios::binary);
	if (!fy) {
		Logger::logFatal(tlFile + " doesn't exist");
		exit(EXIT_FAILURE);
	}
}

std::string UnifiedBinarySampleReader::getFileExt(const std::string& s) {
	size_t i = s.rfind('.', s.length());
	if (i != std::string::npos) {
		return (s.substr(i + 1, s.length() - i));
	}

	return ("");
}

BinFileType UnifiedBinarySampleReader::lookupFileType(const std::string& s) {
	if (s.compare("bin") == 0) {
		return FLOAT;
	}
	if (s.compare("bin8") == 0) {
		return CHAR;
	}
	if (s.compare("bin32") == 0) {
		return INT;
	}
	Logger::logFatal("Wrong files extension");
	exit(EXIT_FAILURE);
}

/**
 * Read a chosen number of samples into matrix X and the corresponding labels
 * into matrix Y.
 */
void UnifiedBinarySampleReader::readLabelledSamples(const size_t batchSize,
		float* X, float* Y) {
	std::vector<size_t> idx(batchSize);
	for (size_t i = 0; i < batchSize; ++i) {
		idx[i] = rr.getLong() % numSamples;
	}
	std::sort(idx.begin(), idx.end());

	retrieveData(batchSize, idx, X, Y);
}

void UnifiedBinarySampleReader::retrieveData(const size_t batchSize,
		const std::vector<size_t>& idx, float* X, float* Y) {
	switch (tdFT) {
	case FLOAT: {
		readRecordsFromBinMat(X, batchSize, idx, sizePerSample, tdFile);
		break;
	}

	case CHAR: {
		uint8* tempX = new uint8[batchSize * sizePerSample];
		readRecordsFromBinMat(tempX, batchSize, idx, sizePerSample, tdFile);
		for (size_t i = 0; i < batchSize * sizePerSample; ++i) {
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
		readRecordsFromBinMat(Y, batchSize, idx, sizePerLabel, tlFile);
		break;
	}

	case CHAR: {
		uint8* tempY = new uint8[batchSize * sizePerLabel];
		readRecordsFromBinMat(tempY, batchSize, idx, sizePerLabel, tlFile);
		for (size_t i = 0; i < batchSize * sizePerLabel; ++i) {
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

} /* namespace rudra */
