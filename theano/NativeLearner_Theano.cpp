#include "NativeLearner_Theano.h"
#include <rudra/NativeLearner.h>
#include <rudra/MLPparams.h>
#include <rudra/io/GPFSSampleClient.h>
#include <rudra/io/UnifiedBinarySampleReader.h>
#include <rudra/io/UnifiedBinarySampleSeqReader.h>
#include <rudra/util/MatrixContainer.h>
#include <rudra/util/RudraRand.h>
#include <dlfcn.h>
#include <iostream>

/**
 * Learner implementation using Theano
 */
namespace rudra {

// begin static methods setting fields in MLPparams
void NativeLearner::setLoggingLevel(int level) {
	// TODO
}

void NativeLearner::setAdaDeltaParams(float rho, float epsilon, float drho,
		float depsilon) {
	if (rho != drho)
		MLPparams::_adaDeltaRho = rho;
	if (epsilon != depsilon)
		MLPparams::_adaDeltaEpsilon = epsilon;
}
void NativeLearner::setMeanFile(std::string fn) {
	MLPparams::_meanFile = fn;
	//    NativeLearner::setMoM(0.0);
	//    NativeLearner::setLRMult(2);
}

void NativeLearner::setSeed(long id, int seed, int defaultSeed) {
	if (seed != defaultSeed) {
		srand(seed);
	} else {
		struct timeval start;
		gettimeofday(&start, NULL);
		unsigned int sd = (unsigned int) ((float) start.tv_usec
				/ (float) ((id + 1) * (id + 1)));
		srand(sd);
	}
}

void NativeLearner::setMoM(float mom) {
	// TODO
}

void NativeLearner::setJobID(std::string jobID) {
	rudra::MLPparams::setWorkingDirectory(jobID);
}

void NativeLearner::initFromCFGFile(std::string confName) {
	MLPparams::initMLPparams(confName);
}
// end statics

NativeLearner::NativeLearner(long id) :
		pid(id), pimpl_(new NativeLearnerImpl()) {
}

void NativeLearner::cleanup() {
	delete pimpl_;
}

NativeLearnerImpl::NativeLearnerImpl() :
		trainSC(NULL), learner_handle(NULL) {
}

NativeLearnerImpl::~NativeLearnerImpl() {
	// TODO: destroy the learner here
	if (learner_handle)
		dlclose(learner_handle);
	if (trainSC != NULL)
		delete trainSC;
}

// write the output per checkptinterval or at the end of job 
void NativeLearner::checkpoint(std::string outputFileName) {
// TODO
	/*
	pimpl_->nn->writeParamsH5(outputFileName);
	if (pimpl_->psu->solverType == rudra::ADAGRAD)
		pimpl_->psu->chkptAdaGrad("ada" + outputFileName);
	 */
}

/**
 * init as learner agent
 */
void NativeLearner::initAsLearner(std::string trainData,
		std::string trainLabels, size_t batchSize, std::string weightsFile,
		std::string solverType) {
	pimpl_->initNetwork(weightsFile);

	// initialize GPFS Sample client for reading training and test data
	char agentName[21];
	sprintf(agentName, "LearnerAgent %6ld", pid);
	pimpl_->trainSC = new GPFSSampleClient(std::string(agentName), batchSize,
			false,
			new UnifiedBinarySampleReader(trainData, trainLabels,
					RudraRand(pid, pid)));
}

void NativeLearner::initAsTester(long placeID, std::string solverType) {
	pimpl_->initNetwork("");
}

/**
 * initialize the network
 */
int NativeLearnerImpl::initNetwork(std::string weightsFile) {
	// For now use a constant, make this configurable later on.
	learner_handle = dlopen("theanolearner.so", RTLD_LAZY | RTLD_LOCAL);

	if (learner_handle == NULL) {
		std::cerr << "Error loading learner: " << dlerror() << std::endl;
		// TODO throw error?  I'm not sure but we need to indicate that
		// there was an error and the object is not useable.
	}

// Probably would have to throw an appropriate error
#define LOAD_SYM(name) do {                                             \
      char *err;                                                        \
      dlerror();                                                        \
      name = reinterpret_cast<name ## _t *>(dlsym(learner_handle, #name)); \
      err = dlerror();                                                  \
      if (err != NULL) {                                                \
        std::cerr << "Error loading symbol " #name ": "<< err << std::endl; \
      }                                                                 \
    } while (0)

	LOAD_SYM(learner_init);
	LOAD_SYM(learner_destroy);
	LOAD_SYM(learner_netsize);
	LOAD_SYM(learner_train);
	LOAD_SYM(learner_test);
	LOAD_SYM(learner_getgrads);
	LOAD_SYM(learner_accgrads);
	LOAD_SYM(learner_updatelr);
	LOAD_SYM(learner_getweights);
	LOAD_SYM(learner_setweights);
	LOAD_SYM(learner_updweights);

#undef LOAD_SYM

	struct param *params = new struct param[MLPparams::MLPCfg.size()];
	size_t i = 0;
	int res;

	if (params == NULL) {
		std::cerr << "Could not allocate network parameter buffer" << std::endl;
		return 1;
	}

	for (keyMap::const_iterator it = MLPparams::MLPCfg.begin();
			it != MLPparams::MLPCfg.end(); ++it, ++i) {
		params[i].key = it->first.c_str();
		params[i].val = it->second.c_str();
		std::cout << "Param: " << i << " " << params[i].key << " "
				<< params[i].val << std::endl;
	}

	res = learner_init(&learner_data, params, MLPparams::MLPCfg.size());

	delete[] params;
	return res;
}

int NativeLearner::getNetworkSize() {
	return pimpl_->learner_netsize(pimpl_->learner_data);
}

float NativeLearner::trainMiniBatch() {
	MatrixContainer<float> minibatchX(pimpl_->trainSC->getSizePerSample(),
			pimpl_->trainSC->batchSize);
	MatrixContainer<float> minibatchY(pimpl_->trainSC->getSizePerLabel(),
			pimpl_->trainSC->batchSize);
	float trainMBErr = pimpl_->learner_train(pimpl_->learner_data,
			pimpl_->trainSC->batchSize, minibatchX.buf,
			pimpl_->trainSC->getSizePerSample(), minibatchY.buf,
			pimpl_->trainSC->getSizePerLabel());
	return trainMBErr;
}

void NativeLearner::getGradients(float *gradients) {
	pimpl_->learner_getgrads(pimpl_->learner_data, gradients);
}

void NativeLearner::accumulateGradients(float *gradients) {
	pimpl_->learner_accgrads(pimpl_->learner_data, gradients);
}

void NativeLearner::setLearningRateMultiplier(float lrMult) {
	pimpl_->learner_updatelr(pimpl_->learner_data, lrMult);
}

void NativeLearner::serializeWeights(float *weights) {
	pimpl_->learner_getweights(pimpl_->learner_data, weights);
}

void NativeLearner::deserializeWeights(float *weights) {
	pimpl_->learner_setweights(pimpl_->learner_data, weights);
}

/**
 * assuming this call is always 


 #define UU_NOP_FLAG  1 // just plain sum update
 #define UU_SMB_FLAG 2 // finish a "super mb" (i.e., super mini-batch size)
 #define UU_EPOCH_FLAG 4 // finish the entire epoch (now the time to send a test to test server)
 */

/**
 *
 */
void NativeLearner::acceptGradients(float *gradients, size_t numMB) {
	// This code needs to accept gradients generated remotely and use them
	// to update the weights.
	// TODO: This is where the update rule, e.g. adagrad is important.
	pimpl_->learner_updweights(pimpl_->learner_data, gradients, numMB);
}

float NativeLearner::testOneEpochSC(float *weights, size_t numTesters, size_t myIndex) {
	char agentName[21];
	sprintf(agentName, "TestClient %6ld", pid);
	size_t batchSize = std::min(rudra::MLPparams::_numTestSamples,
			rudra::MLPparams::_batchSize);
	size_t numMB = std::max((size_t) 1,
			rudra::MLPparams::_numTestSamples / batchSize);
	size_t mbPerLearner = std::max((size_t) 1, numMB / numTesters);
	size_t startMB = myIndex * mbPerLearner;
	float totalTestErr = 0.0f;
	if (startMB < numMB) {
		size_t cursor = startMB * rudra::MLPparams::_batchSize;
		rudra::GPFSSampleClient testSC(std::string(agentName),
				MLPparams::_batchSize, true,
				new rudra::UnifiedBinarySampleSeqReader(
						rudra::MLPparams::_testData,
						rudra::MLPparams::_testLabels, cursor));
		deserializeWeights(weights);
		MatrixContainer<float> minibatchX(rudra::MLPparams::_batchSize,
				rudra::MLPparams::_numInputDim);
		MatrixContainer<float> minibatchY(rudra::MLPparams::_batchSize,
				testSC.getSizePerLabel());
		for (size_t i = 0; i < mbPerLearner; i++) {
			testSC.getLabelledSamples(minibatchX.buf, minibatchY.buf);
			totalTestErr += pimpl_->learner_test(pimpl_->learner_data,
					testSC.batchSize, minibatchX.buf, testSC.getSizePerSample(),
					minibatchY.buf, testSC.getSizePerLabel());
		}
	}
	float testError = totalTestErr / mbPerLearner;
	// cosmetic changes, to return a percentage, instead of a fraction
	return testError * 100;
}

} // namespace rudra

