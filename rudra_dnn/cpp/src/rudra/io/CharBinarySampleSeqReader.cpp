/*
 * CharBinarySampleSeqReader.cpp
 */

#include "rudra/io/CharBinarySampleSeqReader.h"
#include "rudra/util/MatrixContainer.h"
#include "rudra/MLPparams.h"
#include <vector>
#include <algorithm>

namespace rudra {
CharBinarySampleSeqReader::CharBinarySampleSeqReader(std::string sampleFileName,
					     std::string labelFileName, size_t totalSampleNum) :
		sizePerImg(MLPparams::_numInputDim), tdFile(sampleFileName), tlFile(
										    labelFileName), totalSampleNum(totalSampleNum),cursor(0) {
    this->setLabelDim();
    this->sizePerLabel = MLPparams::_labelDim;
}

CharBinarySampleSeqReader::CharBinarySampleSeqReader(std::string sampleFileName,
					     std::string labelFileName, size_t totalSampleNum, size_t cursor) :
		sizePerImg(MLPparams::_numInputDim), tdFile(sampleFileName), tlFile(
										    labelFileName), totalSampleNum(totalSampleNum),cursor(cursor) {
    this->setLabelDim();
    this->sizePerLabel = MLPparams::_labelDim;
}

CharBinarySampleSeqReader::~CharBinarySampleSeqReader() {
}

/**
 * Read a chosen number of samples into matrix X and the corresponding labels
 * into matrix Y.
 */
void CharBinarySampleSeqReader::readLabelledSamples(size_t numSamples, MatrixContainer<float> &X, MatrixContainer<float>& Y) {
	std::vector<size_t> idx(numSamples);
	for (size_t i = 0; i < numSamples; ++i) {
	    idx[i] = (cursor++) % totalSampleNum;
	}
	
	// binary format is transposed
	MatrixContainer<uint8> tX(MLPparams::_batchSize, MLPparams::_numInputDim);
	MatrixContainer<float> tY(MLPparams::_batchSize, MLPparams::_labelDim); // May 15, 2015, labelDim
	 // std::cout<<"tX.dimM: "<<tX.dimM<<" tX.dimN: "<<tX.dimN<<std::endl;
	 // std::cout<<"tY.dimN: "<<tY.dimM<<" tY.dimN: "<<tY.dimN<<std::endl;
	for (size_t i = 0; i < numSamples; ++i) {
	    readBinMat<uint8>(tX.buf + i * sizePerImg, idx[i], 1, sizePerImg, tdFile);
	    readBinMat<float>(tY.buf + i * sizePerLabel, idx[i], 1, sizePerLabel, tlFile); // label file always read as floats
	}
	X = tX;
	Y = tY;

//	tX.transposeTo(X);
//	tY.transposeTo(Y);
}

  
    void CharBinarySampleSeqReader::setLabelDim(){
	int rows, columns;
        MLPparams::readBinHeader(tlFile, rows, columns);
	MLPparams::setLabelDim(columns);
    }
   

} /* namespace rudra */
