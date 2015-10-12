/*
 * UnifiedBinarySampleSeqReader.cpp
 *
 *  Created on: Jul 14, 2015
 *      Author: weiz
 */

#include "rudra/io/UnifiedBinarySampleSeqReader.h"
#include <vector>

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
 * Read a chosen number of samples into buffer X and the corresponding labels
 * into buffer Y.
 */
void UnifiedBinarySampleSeqReader::readLabelledSamples(const size_t numSamples,
		float* X, float* Y) {

    // prepare the indices that we need to retrieve
	std::vector<size_t> idx(numSamples);
	for (size_t i = 0; i < numSamples; ++i) {
	    idx[i] = (cursor++) % totalSampleNum;
	}
	retrieveData(numSamples, idx, X, Y);
}

} /* namespace rudra */
