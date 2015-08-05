/*
 * BinarySampleReader.h
 */

#ifndef RUDRA_IO_BINARYSAMPLEREADER_H_
#define RUDRA_IO_BINARYSAMPLEREADER_H_

#include "rudra/io/SampleReader.h"
#include "rudra/util/MatrixContainer.h"
#include "rudra/util/RudraRand.h"
#include <cstdlib>
namespace rudra {
class BinarySampleReader: public SampleReader {
public:
	const size_t sizePerImg;
	size_t sizePerLabel;
       
	std::string tdFile; // training data file, GPFS can hold large enough data
	std::string tlFile; // training label file
	RudraRand rr;
	BinarySampleReader(std::string sampleFileName, std::string labelFileName, RudraRand rr);
	~BinarySampleReader();

	void readLabelledSamples(size_t numSamples, MatrixContainer<float> &X,
			MatrixContainer<float>& Y);
	void setLabelDim();

};
} /* namespace rudra */

#endif /* RUDRA_IO_BINARYSAMPLEREADER_H_ */
