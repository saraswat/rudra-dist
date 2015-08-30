/*
 * SampleClient.h
 *      This is purely an abstract class
 */

#ifndef RUDRA_SAMPLE_SAMPLECLIENT_H_
#define RUDRA_SAMPLE_SAMPLECLIENT_H_


#include "rudra/util/MatrixContainer.h"
#include <cstddef>
#define SC_NULL 0 // not using SampleClient
#define MPI_SC 1
#define GPFS_SC 2
#define GPFS_THREADED_SC 3 // multi-threaded gpfs client
#define GPFS_MF 4 // multi-file for training data and test data, treat trainData as the file that contains a list
                  // training data files and etc 
namespace rudra {
  template<class T> class MatrixContainer;
class SampleClient {
public:
	virtual void getLabelledSamples(MatrixContainer<float> &samples,
			MatrixContainer<float> &labels) = 0;
};
} /* namespace rudra */

#endif /* RUDRA_SAMPLE_SAMPLECLIENT_H_ */
