/*
 * BinarySampleMFReader.h, MF stands for Multiple File
 */

#ifndef RUDRA_IO_BINARYSAMPLE_MF_READER_H_
#define RUDRA_IO_BINARYSAMPLE_MF_READER_H_

#include "rudra/io/SampleReader.h"
#include "rudra/util/MatrixContainer.h"
#include "rudra/util/RudraRand.h"
#include <map>
namespace rudra {
class BinarySampleMFReader: public SampleReader {
public:
	const size_t sizePerImg;
	size_t sizePerLabel;

	std::string tdSummaryFile; // training data summary file, which contains a list of training data files
	std::string tlSummaryFile; // training label summar file, which contains a list of training label files

	BinarySampleMFReader(std::string sampleSummaryFileName, std::string labelSummaryFileName, size_t totalSampleNum, RudraRand rr);
	~BinarySampleMFReader();

	void readLabelledSamples(size_t numSamples, MatrixContainer<float> &X,
			MatrixContainer<float>& Y);
	
	void initXYTabs();
	
	void checkTabConsistency();
	std::map<size_t, std::string> xTab; // training data look up table
	std::map<size_t, std::string> yTab; // label look up table
	size_t xSize; // total number of samples in xTab
	size_t ySize; // total number of samples in yTab
	size_t totalSampleNum; // total number of samples, xSize == ySize == totalSampleNum
	RudraRand rr;
	void setLabelDim();


	void populateTable(std::string fileName, std::map<size_t, std::string> &tab, size_t &size);
	void lookupTab(std::map<size_t, std::string> tab, size_t idx, std::string &fName, size_t &fIdx);
	

};
} /* namespace rudra */

#endif /* RUDRA_IO_BINARYSAMPLEREADER_H_ */
