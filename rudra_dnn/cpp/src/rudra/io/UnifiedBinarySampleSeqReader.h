/*
 * UnifiedBinarySampleSeqReader.h
 *
 *  Created on: Jul 14, 2015
 *      Author: weiz
 */

#ifndef UNIFIEDBINARYSAMPLESEQREADER_H_
#define UNIFIEDBINARYSAMPLESEQREADER_H_

#include "rudra/io/UnifiedBinarySampleReader.h"
#include "rudra/util/MatrixContainer.h"
// added on May 19, 2015, to support bin reader for sequential reading
namespace rudra {
class UnifiedBinarySampleSeqReader: public UnifiedBinarySampleReader {
public:
	/*const size_t sizePerImg;
	size_t sizePerLabel;

	std::string tdFile; // most likely test data file, GPFS can hold large enough data
	std::string tlFile; // most likely test label file*/
	size_t totalSampleNum;
	size_t cursor;
	UnifiedBinarySampleSeqReader(std::string sampleFileName, std::string labelFileName, size_t totalSampleNum);
	UnifiedBinarySampleSeqReader(std::string sampleFileName, std::string labelFileName, size_t totalSampleNum, size_t cursor);
	~UnifiedBinarySampleSeqReader();

	void readLabelledSamples(size_t numSamples, MatrixContainer<float> &X,
			MatrixContainer<float>& Y);

	void setLabelDim();

};
} /* namespace rudra */



#endif /* UNIFIEDBINARYSAMPLESEQREADER_H_ */
