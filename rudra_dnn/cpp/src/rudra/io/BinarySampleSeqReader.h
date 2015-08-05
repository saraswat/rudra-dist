/*
 * BinarySampleReader.h
 */

#ifndef RUDRA_IO_BINARYSAMPLE_SEQ_READER_H_
#define RUDRA_IO_BINARYSAMPLE_SEQ_READER_H_

#include "rudra/io/SampleReader.h"
#include "rudra/util/MatrixContainer.h"
// added on May 19, 2015, to support bin reader for sequential reading 
namespace rudra {
class BinarySampleSeqReader: public SampleReader {
public:
	const size_t sizePerImg;
	size_t sizePerLabel;

	std::string tdFile; // most likely test data file, GPFS can hold large enough data
	std::string tlFile; // most likely test label file
	size_t totalSampleNum;
	size_t cursor;
	BinarySampleSeqReader(std::string sampleFileName, std::string labelFileName, size_t totalSampleNum);
	BinarySampleSeqReader(std::string sampleFileName, std::string labelFileName, size_t totalSampleNum, size_t cursor);
	~BinarySampleSeqReader();

	void readLabelledSamples(size_t numSamples, MatrixContainer<float> &X,
			MatrixContainer<float>& Y);
	
	void setLabelDim();

};
} /* namespace rudra */

#endif /* RUDRA_IO_BINARYSAMPLE_SEQ_READER_H_ */
