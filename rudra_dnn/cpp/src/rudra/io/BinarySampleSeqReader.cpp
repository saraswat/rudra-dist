/*
 * BinarySampleSeqReader.cpp
 */

#include "rudra/io/BinarySampleSeqReader.h"
#include "rudra/util/MatrixContainer.h"
#include "rudra/MLPparams.h"
#include <vector>
#include <algorithm>

namespace rudra {
BinarySampleSeqReader::BinarySampleSeqReader(std::string sampleFileName,
					     std::string labelFileName, size_t totalSampleNum) :
		sizePerImg(MLPparams::_numInputDim), tdFile(sampleFileName), tlFile(
										    labelFileName), totalSampleNum(totalSampleNum),cursor(0) {
    this->setLabelDim();
    this->sizePerLabel = MLPparams::_labelDim;
}

BinarySampleSeqReader::BinarySampleSeqReader(std::string sampleFileName,
					     std::string labelFileName, size_t totalSampleNum, size_t cursor) :
		sizePerImg(MLPparams::_numInputDim), tdFile(sampleFileName), tlFile(
										    labelFileName), totalSampleNum(totalSampleNum),cursor(cursor) {
    this->setLabelDim();
    this->sizePerLabel = MLPparams::_labelDim;
}

BinarySampleSeqReader::~BinarySampleSeqReader() {
}

/**
 * Read a chosen number of samples into matrix X and the corresponding labels
 * into matrix Y.
 */
void BinarySampleSeqReader::readLabelledSamples(size_t numSamples, MatrixContainer<float> &X, MatrixContainer<float>& Y) {
	std::vector<size_t> idx(numSamples);
	for (size_t i = 0; i < numSamples; ++i) {
	    idx[i] = (cursor++) % totalSampleNum;
	}
	
	// binary format is transposed
	MatrixContainer<float> tX(MLPparams::_batchSize, MLPparams::_numInputDim);
	MatrixContainer<float> tY(MLPparams::_batchSize, MLPparams::_labelDim); // May 15, 2015, labelDim
	 // std::cout<<"tX.dimM: "<<tX.dimM<<" tX.dimN: "<<tX.dimN<<std::endl;
	 // std::cout<<"tY.dimN: "<<tY.dimM<<" tY.dimN: "<<tY.dimN<<std::endl;
	for (size_t i = 0; i < numSamples; ++i) {

	    readBinMat<float>(tX.buf + i * sizePerImg, idx[i], 1, sizePerImg, tdFile);

	    readBinMat<float>(tY.buf + i * sizePerLabel, idx[i], 1, sizePerLabel, tlFile);

	}
	X = tX;
	Y = tY;

//	tX.transposeTo(X);
//	tY.transposeTo(Y);
}

  
    void BinarySampleSeqReader::setLabelDim(){
	int rows, columns;
        MLPparams::readBinHeader(tlFile, rows, columns);
	MLPparams::setLabelDim(columns);
    }
   

} /* namespace rudra */
