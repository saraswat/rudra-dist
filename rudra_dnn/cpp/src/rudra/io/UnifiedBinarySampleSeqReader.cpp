 /*
 * UnifiedBinarySampleSeqReader.cpp
 *
 *  Created on: Jul 14, 2015
 *      Author: weiz
 */

#include "rudra/io/UnifiedBinarySampleSeqReader.h"
#include "rudra/io/BinaryMatrixReader.h"
#include "rudra/util/MatrixContainer.h"
#include "rudra/MLPparams.h"
#include "rudra/util/Logger.h"
#include <vector>
#include <algorithm>


namespace rudra {
UnifiedBinarySampleSeqReader::UnifiedBinarySampleSeqReader(std::string sampleFileName,
					     std::string labelFileName, size_t totalSampleNum) :
		UnifiedBinarySampleReader(sampleFileName, labelFileName, RudraRand()),
		totalSampleNum(totalSampleNum),cursor(0) {

}

UnifiedBinarySampleSeqReader::UnifiedBinarySampleSeqReader(std::string sampleFileName,
					     std::string labelFileName, size_t totalSampleNum, size_t cursor) :
		UnifiedBinarySampleReader(sampleFileName, labelFileName, RudraRand()),
		totalSampleNum(totalSampleNum), cursor(cursor) {

}

UnifiedBinarySampleSeqReader::~UnifiedBinarySampleSeqReader() {
}

/**
 * Read a chosen number of samples into matrix X and the corresponding labels
 * into matrix Y.
 */
void UnifiedBinarySampleSeqReader::readLabelledSamples(size_t numSamples, MatrixContainer<float> &X, MatrixContainer<float>& Y) {

    // step 1 prepare the indices that we need to retrieve
	std::vector<size_t> idx(numSamples);
	for (size_t i = 0; i < numSamples; ++i) {
	    idx[i] = (cursor++) % totalSampleNum;
	}


	// step 2 , retrieve data
	switch(tdFT){
		case FLOAT:{
			assert(X.dimM >= numSamples);
			assert(X.dimN == MLPparams::_numInputDim);
			readRecordsFromBinMat(X.buf, numSamples, idx, sizePerImg, tdFile);
			break;
		}

		case CHAR:{
			MatrixContainer<uint8> tempX(MLPparams::_batchSize, MLPparams::_numInputDim);
			readRecordsFromBinMat(tempX.buf, numSamples, idx, sizePerImg, tdFile);
			X = tempX; // convert from uint8 to float
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
			assert(Y.dimM >= numSamples);
			assert(Y.dimN == MLPparams::_labelDim);
			readRecordsFromBinMat(Y.buf, numSamples, idx, sizePerLabel, tlFile);
			break;
		}

		case CHAR:{
			assert(Y.dimM >= numSamples);
			assert(Y.dimN == MLPparams::_labelDim);
			MatrixContainer<uint8> tempY(MLPparams::_batchSize, MLPparams::_labelDim);
			readRecordsFromBinMat(tempY.buf, numSamples, idx, sizePerLabel, tlFile);
			Y = tempY; // convert from uint8 to float
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





} /* namespace rudra */
