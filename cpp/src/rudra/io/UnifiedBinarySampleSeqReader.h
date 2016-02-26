/*
 * UnifiedBinarySampleSeqReader.h
 */

#ifndef UNIFIEDBINARYSAMPLESEQREADER_H_
#define UNIFIEDBINARYSAMPLESEQREADER_H_

#include "rudra/io/UnifiedBinarySampleReader.h"

namespace rudra {
class UnifiedBinarySampleSeqReader: public UnifiedBinarySampleReader {
public:
	size_t cursor;
	UnifiedBinarySampleSeqReader(std::string sampleFileName,
			std::string labelFileName);
	UnifiedBinarySampleSeqReader(std::string sampleFileName,
			std::string labelFileName, size_t cursor);
	~UnifiedBinarySampleSeqReader();

	void readLabelledSamples(size_t batchSize, float* X, float* Y);

	void setLabelDim();
};
} /* namespace rudra */

#endif /* UNIFIEDBINARYSAMPLESEQREADER_H_ */
