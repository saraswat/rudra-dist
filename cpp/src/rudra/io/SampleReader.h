/*
 * SampleReader.h
 *  *      This is purely an abstract class
 */

#ifndef RUDRA_IO_SAMPLEREADER_H_
#define RUDRA_IO_SAMPLEREADER_H_

#include <cstddef>

namespace rudra {
class SampleReader {
public:
	const size_t sizePerSample;
	size_t sizePerLabel; // should be const
	SampleReader(const size_t sizePerSample) :
			sizePerSample(sizePerSample) {
	}

	virtual void readLabelledSamples(const size_t numSamples, float* X, float* Y) = 0;
};
}

#endif /* RUDRA_IO_SAMPLEREADER_H_ */
