/*
 * UnifiedBinarySampleReader.h
 *
 *  Created on: Jul 13, 2015
 *      Author: weiz
 */

#ifndef UNIFIEDBINARYSAMPLEREADER_H_
#define UNIFIEDBINARYSAMPLEREADER_H_

#include "rudra/io/SampleReader.h"
#include "rudra/util/MatrixContainer.h"
#include "rudra/util/RudraRand.h"
#include <cstdlib>
namespace rudra {
enum BinFileType {CHAR, INT, FLOAT, INVALID};
class UnifiedBinarySampleReader: public SampleReader {
public:
	const size_t sizePerImg;
	size_t sizePerLabel;

	std::string tdFile; // training data file, GPFS can hold large enough data
	std::string tlFile; // training label file
	BinFileType tdFT; // training data file type, added July 13, 2015
	BinFileType tlFT; // training label file type, added July 13, 2015
	RudraRand rr;
	UnifiedBinarySampleReader(std::string sampleFileName, std::string labelFileName, RudraRand rr);
	~UnifiedBinarySampleReader();
	void checkFiles();// to check if files exist
	void initFileTypes();
	std::string getFileExt(const std::string& s);
	BinFileType lookupFileType(const std::string& s);
	void readLabelledSamples(size_t numSamples, MatrixContainer<float> &X,
			MatrixContainer<float>& Y);
	void setLabelDim();

};
} /* namespace rudra */



#endif /* UNIFIEDBINARYSAMPLEREADER_H_ */
