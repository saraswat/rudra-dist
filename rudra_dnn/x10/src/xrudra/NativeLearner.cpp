#include "NativeLearner.h"
#include <rudra/MLPparams.h>
#include <rudra/io/GPFSSampleClient.h>
#include <rudra/io/UnifiedBinarySampleReader.h>
#include <rudra/io/UnifiedBinarySampleSeqReader.h>
#include <rudra/util/Logger.h>
#include <rudra/util/MatrixContainer.h>
#include <rudra/util/RudraRand.h>
#include <iostream>

// vj -- modifying this for integration with Theano. Originally the code supported the Rudra native learner.
// see NativeLearnerRudra.cpp.aside for the code.

// refactor later
namespace xrudra {

    long NativeLearner::getNumEpochs() {
        return rudra::MLPparams::_numEpochs;
    }

    long NativeLearner::getNumTrainingSamples() {
        return rudra::MLPparams::_numTrainSamples;
    }

    long NativeLearner::getMBSize() {
        return rudra::MLPparams::_batchSize;
    }

  // begin static methods setting fields in MLPparams
  void NativeLearner::setLoggingLevel(int level) {
    rudra::Logger::setLoggingLevel(level);
  }
  void NativeLearner::setJobDir(std::string jobDir) {
    rudra::MLPparams::setJobID(jobDir);
  }

  void NativeLearner::setAdaDeltaParams(float rho, float epsilon,
                                        float drho, float depsilon) {
   if (rho != drho) rudra::MLPparams::_adaDeltaRho = rho;
    if (epsilon != depsilon) rudra::MLPparams::_adaDeltaEpsilon = epsilon;
  }
  void NativeLearner::setMeanFile(std::string fn){
    rudra::MLPparams::_meanFile = fn;
    //    NativeLearner::setMoM(0.0);
    //    NativeLearner::setLRMult(2);
  }

  void NativeLearner::setSeed(long id, int seed, int defaultSeed) {
    if (seed != defaultSeed) {
      rudra::MLPparams::_randSeed=seed;
      srand(seed);
    } else {
      struct timeval start;
      gettimeofday(&start, NULL);
      unsigned int sd = (unsigned int) ((float) start.tv_usec / (float)( (id+1)*(id+1) ));
      srand(sd);
    }
  }

  void NativeLearner::setMoM(float mom) {
    rudra::MLPparams::_mom = mom;    
  }

  void NativeLearner::setLRMult(float mult) {
    rudra::MLPparams::_lrMult = mult;
  }

  void NativeLearner::setWD() {
    rudra::MLPparams::setWD();
  }

  void NativeLearner::initFromCFGFile(std::string confName) {
    rudra::MLPparams::initMLPparams(confName);
  }
  // end statics

  NativeLearner::NativeLearner(long id) :
    pid(id),
    trainSC(NULL),
    learner_handle(NULL)
  {}

    void NativeLearner::cleanup() {
    // TODO: destroy the learner here
    if (learner_handle)
      dlclose(learner_handle);
        if (trainSC != NULL) delete trainSC;
    }

// write the output per checkptinterval or at the end of job 
  void NativeLearner::checkpointIfNeeded(int whichEpoch) {
// TODO
/*
    int chkptInterval = rudra::MLPparams::_chkptInterval;
    int epochNum = rudra::MLPparams::_numEpochs;
    std::string jobID = rudra::MLPparams::_jobID;

    if ((chkptInterval != 0 && whichEpoch % chkptInterval == 0) || whichEpoch == epochNum) {
      std::string outputFileName = "";
      if (whichEpoch == epochNum) {
        outputFileName = jobID + ".final" + ".h5";
      } else {
        std::stringstream ss;
        ss << whichEpoch;
        outputFileName = jobID + ".epoch." + ss.str() + ".h5";
      }
      std::cout << "[Tester] Writing weights to " << outputFileName << std::endl;
      nn->writeParamsH5(outputFileName);
      if (psu->solverType==rudra::ADAGRAD)
        psu->chkptAdaGrad("ada"+outputFileName);
    }
*/
  }

/**
 * init as learner agent
 */
void NativeLearner::initAsLearner(std::string weightsFile, std::string solverType) {
    initNetwork(weightsFile);

    // initialize GPFS Sample client for reading training and test data
    char agentName[21];
    sprintf(agentName, "LearnerAgent %6ld", pid);
    trainSC = new rudra::GPFSSampleClient(std::string(agentName), false,
        new rudra::UnifiedBinarySampleReader(rudra::MLPparams::_trainData, 
                                             rudra::MLPparams::_trainLabels, 
                                             rudra::RudraRand(pid, pid)));

}

void NativeLearner::initAsTester(long placeID, std::string solverType) {
    initNetwork("");
}

/**
 * initialize the network
 */
  int NativeLearner::initNetwork(std::string weightsFile){
    // For now use a constant, make this configurable later on.
    learner_handle = dlopen("theano", RTLD_LAZY|RTLD_LOCAL);

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

  struct param *params = new struct param[rudra::MLPparams::MLPCfg.size()];
  size_t i = 0;
  int res;

  if(params == NULL){
    std::cerr<<"Could not allocate network parameter buffer"<<std::endl;
    return 1;
  }

  for(keyMap::const_iterator it = rudra::MLPparams::MLPCfg.begin();
      it != rudra::MLPparams::MLPCfg.end(); ++it, ++i){
    params[i].key = it->first.c_str();
    params[i].val = it->second.c_str();
  }

  res = learner_init(&learner_data, params, rudra::MLPparams::MLPCfg.size());

  delete[] params;
  return res;
  }

int NativeLearner::getNetworkSize() {
  return learner_netsize(learner_data);
}

float NativeLearner::trainMiniBatch() {
    rudra::MatrixContainer<float> minibatchX(rudra::MLPparams::_numInputDim, rudra::MLPparams::_batchSize);
    rudra::MatrixContainer<float> minibatchY(rudra::MLPparams::_numClasses, rudra::MLPparams::_batchSize);
    float trainMBErr = learner_train(learner_data, rudra::MLPparams::_batchSize,
                                minibatchX.buf, rudra::MLPparams::_numInputDim,
                                minibatchY.buf, rudra::MLPparams::_numClasses);
    return trainMBErr;
}

void NativeLearner::getGradients(float *gradients) {
    learner_getgrads(learner_data, gradients);
}

void NativeLearner::accumulateGradients(float *gradients) {
    learner_accgrads(learner_data, gradients);
}

/**
 * update learning rate
 */
void NativeLearner::updateLearningRate(long curEpochNum){
// compatible with update learning rate in the new scheme
    std::cout << "NativeLearner::updateLearningRate pid " << pid
        << " epoch " << curEpochNum << " learning rate " 
        << rudra::MLPparams::_lrMult*rudra::MLPparams::LearningRateMultiplier::_lr[curEpochNum]
        << std::endl;
  learner_updatelr(learner_data, rudra::MLPparams::_lrMult*rudra::MLPparams::LearningRateMultiplier::_lr[curEpochNum]);
}

void NativeLearner::serializeWeights(float *weights){
    learner_getweights(learner_data, weights);
}

void NativeLearner::deserializeWeights(float *weights){
    learner_setweights(learner_data, weights);
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
void NativeLearner::acceptGradients(float *gradients, size_t numMB){
  // This code needs to accept gradients generated remotely and use them
  // to update the weights.
  // TODO: This is where the update rule, e.g. adagrad is important.
  learner_updweights(learner_data, gradients, numMB);
}

float NativeLearner::testOneEpochSC(float *weights, size_t numTesters) {
/*
    char agentName[21];
    sprintf(agentName, "TestClient %6ld", pid);
    size_t batchSize = std::min(rudra::MLPparams::_numTestSamples,
                                rudra::MLPparams::_batchSize);
    size_t numMB = std::max((size_t) 1, rudra::MLPparams::_numTestSamples / batchSize);
    size_t mbPerLearner = std::max((size_t) 1, numMB / numTesters);
    size_t startMB = pid * mbPerLearner;
    size_t numSamplePerLearner = rudra::MLPparams::_numTestSamples / numTesters + 1;
    float totalTestErr = 0.0f;
    if (startMB < numMB) {
        size_t cursor = startMB * rudra::MLPparams::_batchSize;
        rudra::GPFSSampleClient testSC(std::string(agentName), true, 
            new rudra::UnifiedBinarySampleSeqReader(rudra::MLPparams::_testData, 
                                                    rudra::MLPparams::_testLabels, 
                                                    numSamplePerLearner, cursor));
        nn->deserialize(weights);
        rudra::MatrixContainer<float> minibatchX(rudra::MLPparams::_batchSize, rudra::MLPparams::_numInputDim);
        rudra::MatrixContainer<float> minibatchY(rudra::MLPparams::_batchSize, testSC.getSizePerLabel());
        for (size_t i = 0; i < mbPerLearner; i++) {
          testSC.getLabelledSamples(minibatchX.buf, minibatchY.buf);
          totalTestErr += nn->testNetworkMinibatch(minibatchX, minibatchY);
        }
    }
    float testError = totalTestErr / mbPerLearner;
    // cosmetic changes, to return a percentage, instead of a fraction
    return testError * 100;
*/
  // TODO: Arnaud. Need to flesh this out for Theano. This code will instantiate the given network
  // with the given weights, and then run the network on the given tests to report the test score.
  return 0.0;
  }

}


