#ifndef __NATIVE_LEARNER_IMPL_H
#define __NATIVE_LEARNER_IMPL_H

#include <cstdlib>
#include <cstddef>
#include <string>
#include <sys/time.h>

namespace rudra {
class NativeLearnerImpl;

class NativeLearner {

public:
	NativeLearner(long id);
	void cleanup(); // X10 native integration can't call C++ destructors

	long getNumEpochs();
	long getNumTrainingSamples();
	long getMBSize();

	// static methods section. These methods are called from static
	// Learner.initNativeStatics, once in each place.

	/** Set the working directory, for various files to be written out, e.g. weightsFile.*/
	static void setLoggingLevel(int level);
	static void setJobDir(std::string jobDir);
	static void setMeanFile(std::string _fileName);
	static void setAdaDeltaParams(float rho, float epsilon, float drho,
			float depsilon);
	static void setSeed(long id, int seed, int defaultSeed);
	static void setMoM(float f);
	static void setLRMult(float mult);
	static void setWD();
	static void initFromCFGFile(std::string confName);
	// end of static methods

	void initAsLearner(std::string weightsFile, std::string solverType);
	void initAsTester(long placeID, std::string solverType);

	int getNetworkSize();

    /**
     * Train the network with a minibatch of samples, storing the gradients
     * for later use and returning the training error [0.0-1.0]. Does not
     * actually update the weights; the update will be performed by a later
     * call to acceptGradients.
     */
	float trainMiniBatch();

	/**
	 * Copy the most recent set of computed gradients into the array provided.
	 * The gradients array must be of size >= [getNetworkSize()].
	 */
	void getGradients(float *gradients);

	/**
	 * Sum the most recent set of computed gradients into the array provided.
	 * The gradients array must be of size >= [getNetworkSize()], and may
	 * contain previously computed gradients.
	 */
	void accumulateGradients(float *gradients);

	/**
     * Output the parameters now into a file, if instructed by job configuration.
	 * Examines the following parameters in MLPparams: _ckptInterval, _numEpochs,
	 * _jobID. The parameters are printed out if this is the last epoch, or
	 * if this epoch modulo chkptInterval is 0 (if ckptInterval > 0). The name of the
	 * file is jobID.final.h5 or jobId.epoch.<whichEpoch>.h5, and it is placed in
	 * current working dir, which is RUDRA_HOME/LOG/jobID/.
     *
	 * @param whichEpoch -- the current epoch
	 */
	void checkpointIfNeeded(int whichEpoch);

	/**
	 * Copy the current set of weights into the array provided.
	 * The weights array must be of size >= [getNetworkSize()].
	 */
	void serializeWeights(float *weights);

	/**
	 * Replace this learner's weights with the weights provided.
	 * The weights array must be of size >= [getNetworkSize()].
	 */
	void deserializeWeights(float *weights);

	void updateLearningRate(long curEpochNum);

    /**
     * Update the network weights by applying the update rule with
     * the given set of gradients.
     * @param grad the gradients
     * @param numMB the number of minibatches which contributed to gradients
     */
	void acceptGradients(float *grad, size_t numMB);

	/**
	 * Initialize the network with the given weights, score your fraction
	 * of the test data, and return the result.
	 */
	float testOneEpochSC(float *weights, size_t numTesters);

private:
    NativeLearnerImpl* pimpl_;
	long pid;
};
} // namespace rudra
#endif
