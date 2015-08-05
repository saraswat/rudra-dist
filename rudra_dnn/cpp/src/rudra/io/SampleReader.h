/*
 * SampleReader.h
 *  *      This is purely an abstract class
 */

#ifndef RUDRA_IO_SAMPLEREADER_H_
#define RUDRA_IO_SAMPLEREADER_H_

#include "rudra/util/MatrixContainer.h"
#include "rudra/util/defs.h"
#include <cstddef>

namespace rudra {
class SampleReader {
public:
	virtual void readLabelledSamples(size_t numSamples, MatrixContainer<float> &X,
			MatrixContainer<float>& Y) = 0;
	
};
}

#endif /* RUDRA_IO_SAMPLEREADER_H_ */
