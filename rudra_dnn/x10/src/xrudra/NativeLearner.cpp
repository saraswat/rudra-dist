#include "NativeLearner.h"

extern "C" {
#include <theano.h>
}
#include <rudra/io/UnifiedBinarySampleReader.h>
#include <rudra/io/UnifiedBinarySampleSeqReader.h>
#include "rudra/util/Logger.h"
#include <rudra/util/RudraRand.h>
#include <iostream>

// vj -- modifying this for integration with Theano. Originally the code supported the Rudra native learner.
// see NativeLearnerRudra.cpp.aside for the code.

// refactor later
namespace xrudra {
  NativeLearner::NativeLearner() :
    trainMBErr(-1.0f),
    trainSC(NULL),
    testSC(NULL),
    NUM_LEARNER(0),
    NUM_MB_PER_EPOCH(0),
    testErr(-1.0f),
  {}

void NativeLearner::setMeanFile(std::string fn){
    rudra::MLPparams::_meanFile = fn;
    rudra::MLPparams::_mom = 0.0;
    rudra::MLPparams::_lrMult =2;
}

  void NativeLearner::initNativeLand(long id, const char* confName,
                                     long numLearner){
    // step 1 place id, seed random number generator
    pid = id;
    struct timeval start;
    gettimeofday(&start, NULL);
    unsigned int seed = (unsigned int) ((float) start.tv_usec / (float)( (id+1)*(id+1) ));
    srand(seed);
    // step 2 init the configuration
    //    rudra::Logger::setLoggingLevel(INFO);
    cfgFile = std::string(confName);
    rudra::MLPparams::initMLPparams(cfgFile);

    // step 3 init numLearner
    NUM_LEARNER = numLearner;

    // step 4 initialize NUM_MB_PER_EPOCH e.g., ParamServer.cpp:16-23, at most do L+1/L amount of work
    int tmpMBPerEpoch = rudra::MLPparams::_numTrainSamples / rudra::MLPparams::_batchSize;
    int q = tmpMBPerEpoch / numLearner;
    int r = tmpMBPerEpoch % numLearner;
    if(r == 0){
	NUM_MB_PER_EPOCH = tmpMBPerEpoch;
    }else{
	NUM_MB_PER_EPOCH = (q+1) * numLearner;
    }

    // dlopen leaner library here and extract the required function pointers
}
/**
 * added on Aug 12, 2015, to support compatible weights update as in c++ code
 */
void NativeLearner::initPSU(std::string _solverType){
  // vj: For Theano we need to figure out the right way to bring in the code from PSUUtils
  // commenting out for now. TODO: Discuss with Arnaud.
  /*
// step 5, initilized PSU
    if(_solverType.compare("sgd") == 0){
	solverType = rudra::SGD;
    }else if(_solverType.compare("adagrad") == 0){
	solverType = rudra::ADAGRAD;
    }else{
	std::cerr<<"incorrect solver type specified, going to use SGD solver type instead."<<std::endl;
    }
    assert(nn != NULL);
    psu = new rudra::PSUtils(nn, solverType);
  */
}

/**
 * init as learner agent
 */
void NativeLearner::initAsLA(bool isReconciler){
    initNetwork(isReconciler);     // init network
   // added on May 1, 2015 to init minibatch
    initXY();
    initTrainSC();
}

void NativeLearner::initXY(){
    minibatchX = rudra::MatrixContainer<float>(rudra::MLPparams::_numInputDim, rudra::MLPparams::_batchSize);
    minibatchY = rudra::MatrixContainer<float>(rudra::MLPparams::_numClasses, rudra::MLPparams::_batchSize);
}


void NativeLearner::initTrainSC(){
    char agentName[21];
    sprintf(agentName, "LearnerAgent %6d", pid);
    trainSC = new rudra::GPFSSampleClient(std::string(agentName), false,
        new rudra::UnifiedBinarySampleReader(rudra::MLPparams::_trainData, rudra::MLPparams::_trainLabels, rudra::RudraRand(pid, pid)));
}

/**
 * initialize the network
 */
int NativeLearner::initNetwork(bool isReconciler){
  /* vj: This is the code for stitching in the Native Rudra learner
  nn = rudra::Network::readFromConfigFile(rudra::MLPparams::MLPCfg["layerCfgFile"]);
  nn->setMomentum(0.0);
    nn->mulLearningRate(rudra::MLPparams::_lrMult*rudra::MLPparams::LearningRateMultiplier::_lr[0]); // to make sure the mul takes effect to begin with
  if(nn != NULL){
    return 0;
  }else{
    std::cerr<<"failed to initialize the neural network."<<std::endl;
    return 1;
  }
  */
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

  res = leaner_init(&leaner_data, params, rudra::MLPparams::MLPCfg.size());

  delete[] params;
  return res;
}

int NativeLearner::getNetworkSize() {
  return leaner_netsize(leaner_data);
}

/* This doesn't seem to be used anywhere
void ensureLoaded(const float *p) {
  //  if (THEANO_LEARNER==0) {
  theano_set_param_entry((const void*)p);
  //  }
  // else do nothing
}
*/

void NativeLearner::loadMiniBatch(){
    //sc->getLabelledSamples(rudra::MLPparams::_batchSize, minibatchX, minibatchY);
    trainSC->getLabelledSamples(minibatchX, minibatchY);
}

float NativeLearner::trainMiniBatch(){
  /* vj: The code for native Rudra learner.
    trainMBErr = nn->trainNetworkNoUpdate(minibatchX, minibatchY); // who will gc minibatchX and minibatchY ?
    return trainMBErr;
    //std::cout<<"here here trainMBErr "<<trainMBErr<<std::endl;

    // TODO: return the Rail of the updates
    */
  trainMBErr = learner_train(rudra::MLPparams::_batchSize,
                             minibatchX.buf, rudra::MLPparams::_numInputDim,
                             minibatchY.buf, rudra::MLPparams::_numClasses);
  return trainMBErr;
}

void NativeLearner::getGradients(float *updates) {
  learner_getgrads(learner_data, updates);
}

void NativeLearner::accumulateGradients(float *updates) {
  /* Code for native Rudra learner
	nn->accumulateUpdates(updates);
  */
  learner_accgrads(learner_data, updates);
}

/**
 * update learning rate
 */
void NativeLearner::updateLearningRate(long curEpochNum){
  // vj -- this is the native Rudra code.
  //    nn->mulLearningRate(rudra::MLPparams::_lrMult*rudra::MLPparams::LearningRateMultiplier::_lr[curEpochNum]); // compatible with update learning rate in the new scheme
  leaner_updatelr(learner_data, rudra::MLPparams::_lrMult*rudra::MLPparams::LearningRateMultiplier::_lr[curEpochNum]); // compatible with update learning rate in the new scheme
}


//////////////////// methods on the side of param server ///////////////

void NativeLearner::serializeWeights(float *weights){
  // Native Rudra learner code
  //    nn->serialize(weights);
  learner_getweights(learner_data, weights);
}

void NativeLearner::deserializeWeights(float *weights){
  //    nn->deserialize(weights);
  leaner_setweigths(learner_data, weights);
}


/**
 * assuming this call is always
 */

#define UU_NOP_FLAG  1 // just plain sum update
#define UU_SMB_FLAG 2 // finish a "super mb" (i.e., super mini-batch size)
#define UU_EPOCH_FLAG 4 // finish the entire epoch (now the time to send a test to test server)


/**
 *
 */
void NativeLearner::acceptGradients(float *weights, size_t numMB){
  //    psu->applyUpdateAfterSum(grad, numMB);
  // This code needs to accept gradients generated remotely and use them
  // to update the weights.
  // TODO: This is where the update rule, e.g. adagrad is important.
  learner_updweights(learner_data, weights, numMB);
}

  // Needed for the tester.


float NativeLearner::testOneEpoch(float *weights){
  /*
  rudra::Network* nn = rudra::Network::readFromConfigFile(rudra::MLPparams::MLPCfg["layerCfgFile"]);

  //    rudra::Network *nn = new rudra::Network(rudra::MLPparams::MLPCfg["layerCfgFile"],
  //				       rudra::MLPparams::MLPCfg["logDir"] + "log");
      nn->deserialize(weights);
      std::cout<<"[in native land] before testing network, network size: "<<nn->networkSize<<std::endl;
      testErr = nn->testNetwork(testData, testLabels);
      std::cout<<"[in naitve land] after testing network, testErr: "<<testErr<<std::endl;
      delete nn;
      return testErr;
  */
  // TODO: Arnaud. Need to flesh this out for Theano. This code will instantiate the given network
  // with the given weights, and then run the network on the given tests to report the test score.
  return 0.0;
}

float NativeLearner::testOneEpoch(){
  //  testErr = nn->testNetwork(testData, testLabels);
  //  return testErr;
}
  /*
  //rudra::GPFSSampleClient *NativeLearner::testSC = NULL;
void NativeLearner::initTestSC(){
	char agentName[21];
	sprintf(agentName, "TestClient %6d", pid);
	rudra::UnifiedBinarySampleSeqReader *binReader =
				new rudra::UnifiedBinarySampleSeqReader(rudra::MLPparams::_testData, rudra::MLPparams::_testLabels, rudra::MLPparams::_numTestSamples);
	testSC = new rudra::GPFSSampleClient(std::string(agentName), true, binReader);
}

// added on Aug 13, 2015, to support parallel file reading for testing
void NativeLearner::initTestSC(long placeID, size_t numLearner){
	char agentName[21];
	sprintf(agentName, "TestClient %6d", pid);
	size_t numSamplePerLearner = rudra::MLPparams::_numTestSamples / numLearner + 1;
	size_t cursor = placeID * numSamplePerLearner;
	rudra::UnifiedBinarySampleSeqReader *binReader =
	    new rudra::UnifiedBinarySampleSeqReader(rudra::MLPparams::_testData, rudra::MLPparams::_testLabels, numSamplePerLearner, cursor);
	NativeLearner::testSC = new rudra::GPFSSampleClient(std::string(agentName), true, binReader);
}

float NativeLearner::testOneEpochSC(float *weights){
    assert(testSC != NULL );
	size_t batchSize = std::min(rudra::MLPparams::_numTestSamples,
			rudra::MLPparams::_batchSize);
	size_t minibatchSize = rudra::MLPparams::_numTestSamples / batchSize;
	size_t numMB = std::max((size_t) 1, minibatchSize);
	rudra::Matrix<float> testX, testY;
	testErr = 0;
	rudra::Network *nn = rudra::Network::readFromConfigFile(rudra::MLPparams::MLPCfg["layerCfgFile"]);
	      nn->deserialize(weights);
	for (size_t i = 0; i < numMB; i++) {
	  testSC->getLabelledSamples(testX, testY);
		float _testErr = nn->testNetworkMinibatch(testX, testY);
		testErr = (testErr * i + _testErr) / (i + 1); // running average of testErr
	}
	return (testErr * 100); // cosmetic changes, to return a percentage number, instead of a fraction number.
}

    long NativeLearner::getTestNum(){
	return rudra::MLPparams::_numTestSamples;
    }
  */
}

