/*
 * SampleClient.h
 *      This is purely an abstract class
 */

#ifndef RUDRA_SAMPLE_SAMPLECLIENT_H_
#define RUDRA_SAMPLE_SAMPLECLIENT_H_

#define SC_NULL 0 // not using SampleClient
#define SC_UNDEFINED 1 // was MPI sample client
#define GPFS_SC 2
#define GPFS_THREADED_SC 3 // multi-threaded gpfs client
#define GPFS_MF 4 // multi-file for training data and test data, treat trainData as the file that contains a list
                  // training data files and etc 
namespace rudra {

class SampleClient {
public:
	virtual size_t getSizePerLabel() = 0;
	virtual void getLabelledSamples(float* samples,
			float* labels) = 0;
};
} /* namespace rudra */

#endif /* RUDRA_SAMPLE_SAMPLECLIENT_H_ */
