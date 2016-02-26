/*
 * UnifiedBinarySampleReader.h
 */

#ifndef UNIFIEDBINARYSAMPLEREADER_H_
#define UNIFIEDBINARYSAMPLEREADER_H_

#include "rudra/io/SampleReader.h"
#include "rudra/util/RudraRand.h"
#include <string>
#include <vector>

namespace rudra {
enum BinFileType {
	CHAR, INT, FLOAT, INVALID
};
class UnifiedBinarySampleReader: public SampleReader {
public:
	std::string tdFile; // training data file, GPFS can hold large enough data
	std::string tlFile; // training label file
	BinFileType tdFT; // training data file type, added July 13, 2015
	BinFileType tlFT; // training label file type, added July 13, 2015

	UnifiedBinarySampleReader(std::string sampleFileName,
			std::string labelFileName, RudraRand rr);
	~UnifiedBinarySampleReader();

	std::string getFileExt(const std::string& s);
	BinFileType lookupFileType(const std::string& s);
	void readLabelledSamples(const size_t batchSize, float* X, float* Y);

protected:
	void retrieveData(const size_t numSamples, const std::vector<size_t>& idx,
			float* X, float* Y);
private:
	RudraRand rr;

	void checkFiles(); // to check if files exist
	void initSizePerLabel();
};
} /* namespace rudra */

#endif /* UNIFIEDBINARYSAMPLEREADER_H_ */
