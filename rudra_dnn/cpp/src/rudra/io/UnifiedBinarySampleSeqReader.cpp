 /*
 * UnifiedBinarySampleSeqReader.cpp
 *
 *  Created on: Jul 14, 2015
 *      Author: weiz
 */

#include "rudra/io/UnifiedBinarySampleSeqReader.h"
#include "rudra/util/MatrixContainer.h"
#include "rudra/MLPparams.h"
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





} /* namespace rudra */
