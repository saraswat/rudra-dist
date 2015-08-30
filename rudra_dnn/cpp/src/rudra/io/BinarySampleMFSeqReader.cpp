/*
 * BinarySampleMFReader.cpp
 */

#include "rudra/io/BinarySampleMFSeqReader.h"
#include "rudra/io/BinaryMatrixReader.h"
#include "rudra/util/RudraRand.h"
#include "rudra/util/MatrixContainer.h"
#include "rudra/MLPparams.h"
#include <vector>

namespace rudra {
BinarySampleMFSeqReader::BinarySampleMFSeqReader(std::string sampleFileName,
					   std::string labelFileName, size_t totalSampleNum) :
    BinarySampleMFReader(sampleFileName, labelFileName, totalSampleNum, RudraRand()), cursor(1) { // cursor starts with 1  
}

BinarySampleMFSeqReader::~BinarySampleMFSeqReader() {
}

/**
 * Read a chosen number of samples into matrix X and the corresponding labels
 * into matrix Y.
 */
void BinarySampleMFSeqReader::readLabelledSamples(size_t numSamples,
		MatrixContainer<float> &X, MatrixContainer<float>& Y) {
	std::vector<size_t> idx(numSamples);
	for (size_t i = 0; i < numSamples; ++i) {
	    size_t tmp = (cursor++) % totalSampleNum; // we have a contract that idx is between 1 and totalNumSample
	    idx[i] = (tmp) ? tmp:totalSampleNum; 
	    //std::cout<<"[idx] "<<i<<" "<<idx[i]<<std::endl;
	}
	// binary format is transposed
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

  
} /* namespace rudra */
