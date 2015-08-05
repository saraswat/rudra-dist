/*
 * CharBinarySampleReader.h
 */

#ifndef RUDRA_IO_CHARBINARYSAMPLEREADER_H_
#define RUDRA_IO_CHARBINARYSAMPLEREADER_H_

#include "rudra/io/SampleReader.h"
#include "rudra/util/MatrixContainer.h"
#include "rudra/util/RudraRand.h"
#include <cstdlib>
namespace rudra {
class CharBinarySampleReader: public SampleReader {
public:
	const size_t sizePerImg;
	size_t sizePerLabel;
       
	std::string tdFile; // training data file, GPFS can hold large enough data
	std::string tlFile; // training label file
	RudraRand rr;
	CharBinarySampleReader(std::string sampleFileName, std::string labelFileName, RudraRand rr);
	~CharBinarySampleReader();

	void readLabelledSamples(size_t numSamples, MatrixContainer<float> &X,
			MatrixContainer<float>& Y);
	void setLabelDim();

};
} /* namespace rudra */

#endif /* RUDRA_IO_BCHARINARYSAMPLEREADER_H_ */
