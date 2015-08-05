/*
 * SampleReaderFactory.h
 * This class can generate
 *
 *  .bin8
 *  CharBinarySampleReader
 *  CharBinarySampleSeqReader
 *
 *  .bin
 *  BinarySampleReader (float)
 *  BinarySampleSeqReader (float)
 *
 *  .bin32
 *  IntBinarySampleReader
 *  IntBinarySampleSeqReader
 *
 *
 *
 *
 *  Created on: Jul 13, 2015
 *      Author: weiz
 */

#ifndef SAMPLEREADERFACTORY_H_
#define SAMPLEREADERFACTORY_H_

class SampleReaderFactory {
public:
	SampleReaderFactory();
	virtual ~SampleReaderFactory();
};

#endif /* SAMPLEREADERFACTORY_H_ */
