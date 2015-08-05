/*
 * BinarySampleMFSeqReader.h, MF stands for Multiple File, Seq stands for sequentially reading
 * this is a stateful class, i.e., assuming each call to readLabelledSamples will move cursor numSamples.
 */

#ifndef RUDRA_IO_BINARYSAMPLE_MF_SEQ_READER_H_
#define RUDRA_IO_BINARYSAMPLE_MF_SEQ_READER_H_

#include "rudra/io/BinarySampleMFReader.h"
#include "rudra/util/MatrixContainer.h"
#include <map>
namespace rudra {
class BinarySampleMFSeqReader: public BinarySampleMFReader {
public:

	BinarySampleMFSeqReader(std::string sampleSummaryFileName, std::string labelSummaryFileName, size_t totalSampleNum);
	~BinarySampleMFSeqReader();

	void readLabelledSamples(size_t numSamples, MatrixContainer<float> &X,
			MatrixContainer<float>& Y);

    private:
	size_t cursor; 
	

};
} /* namespace rudra */

#endif /* RUDRA_IO_BINARYSAMPLEREADER_H_ */
