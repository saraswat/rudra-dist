#include <iostream>
#include"NativeUtilsImpl.h"
extern "C" {
#include <theano.h>
}
#include<x10/lang/Rail.h>
#include<rudra/MLPparams.h>
/*#include<rudra/Network.h>*/

#include <rudra/util/MatrixContainer.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <rudra/io/GPFSSampleClient.h> // x10-impl always assumes GPFSSampleClient
#include <rudra/io/BinarySampleReader.h>
#include <rudra/io/BinarySampleSeqReader.h>
#include <rudra/util/RudraRand.h>


// rudra::Network *nn = NULL;
long pid;
float trainMBErr = -1.0f;
rudra::MatrixContainer<float> minibatchX;
rudra::MatrixContainer<float> minibatchY;
rudra::MatrixContainer<float> Data;
rudra::MatrixContainer<float> Labels;
rudra::MatrixContainer<float> testData;  
rudra::MatrixContainer<float>  testLabels;
rudra::GPFSSampleClient *trainSC = NULL;

int NUM_LEARNER = 0;
int NUM_MB_PER_EPOCH = 0;

// vj TODO: This value should be set by examining the cfg file and determining
// if it defines "theanoModel"
int THEANO_LEARNER=0; // 0=true, 1=false

::x10::lang::Rail<x10_float>* ptr2Rail(){
	 //std::string ptrName = std::string(name);

	    long numElms = 10;
	    float *ptr = new float[numElms];
	    for(int i = 0; i < numElms; i++){
	    	ptr[i] = (float) (i+1);
	    }
	    // ::x10::lang::Rail<x10_float>* bufX10 = new ::x10::lang::Rail<x10_float>(numElms);
	    ::x10::lang::Rail<x10_float>* bufX10 = ::x10::lang::Rail<x10_float>::_make(numElms);

	    ::x10::lang::rail_copyRaw(ptr, bufX10->raw, sizeof(float)*numElms, false);
	    return bufX10;


}

::x10::lang::Rail<x10_float>* ptr2Rail(float *src, size_t size){
    //printf("src addr %08x\n", src);
    //printf("size %u \n", size);
    ::x10::lang::Rail<x10_float>* bufX10 = ::x10::lang::Rail<x10_float>::_make(size);
    ::x10::lang::rail_copyRaw(src, bufX10->raw, sizeof(float)*size, false);
    return bufX10;
}

// called everywhere (e.g., LA, SS, PS)
void initNativeLand(long id, const char* confName, long numLearner){
    // step 1 place id, seed random number generator
    pid = id;
    struct timeval start;
    gettimeofday(&start, NULL);
    unsigned int seed = (unsigned int) ((float) start.tv_usec / (float)( (id+1)*(id+1) ));
    srand(seed);
    // step 2 init the configuration
    std::string cfgFile = std::string(confName);
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
}


/**
 * init as learner agent
 */
void initAsLA(){
    // init network
    initNetwork();
   // added on May 1, 2015 to init minibatch
    initXY();
    initTrainSC();
}

void initXY(){
    minibatchX = rudra::MatrixContainer<float>(rudra::MLPparams::_numInputDim, rudra::MLPparams::_batchSize);
    minibatchY = rudra::MatrixContainer<float>(rudra::MLPparams::_numClasses, rudra::MLPparams::_batchSize);
}


void initTrainSC(){
    char agentName[21];
    sprintf(agentName, "LearnerAgent %6d", pid);
    std::cout<<"learner agent "<<pid<<" initialize SC"<<std::endl;
    rudra::RudraRand rr(pid, pid); //added on May 28, 2015
    std::cout<<"train data: "<<rudra::MLPparams::_trainData<<" train label "<<rudra::MLPparams::_trainLabels<<std::endl;
    trainSC = new rudra::GPFSSampleClient(agentName, true, new rudra::BinarySampleReader(rudra::MLPparams::_trainData, rudra::MLPparams::_trainLabels, rr));
    
}

/**
 * init as stats server
 */
void initAsSS(){
    // currently do nothing
}

// vj TODO: Remove this
void initAsPS(){
    // init network
    initNetwork();
}

/**
 * initialize the network
 */
int initNetwork(){
  //  if (THEANO_LEARNER==0) {
    // we are going to not set nn here. Maybe later we will see about getting an nn wrapper
    // for the code in Theano.c
    theano_init();
    return 0;
    //  }
  /*vj
    nn = rudra::Network::readFromConfigFile(rudra::MLPparams::MLPCfg["layerCfgFile"]);    
	if(nn != NULL){
	    return 0;
	}else{
	    std::cerr<<"failed to initialize the neural network."<<std::endl;
	    return 1;
	}
  */
}

int getNetworkSize() {
  //  if (THEANO_LEARNER==0) 
    return theano_networkSize();
    //  return nn->networkSize;
}

int loadTrainData(){
    return 0;
}

void ensureLoaded(const  float *p) {
  //  if (THEANO_LEARNER==0) {
  theano_set_param_entry((const void*)p);
  //  }
  // else do nothing
}

// vj I This loads data from the underlying data souce into trainSC.
// this can work as is for Theano.
// after this call, a minibatch of the given shape is loaded into trainSC

void selTrainMB(){
    //sc->getLabelledSamples(MLPparams::_batchSize, minibatchX, minibatchY);
    trainSC->getLabelledSamples(minibatchX, minibatchY);
}

void trainOneMB(){
  //  if (THEANO_LEARNER==0) {
    // At this point MLPparams is already initialized, so one can simply read the value of
    // fdims, tndims
    trainMBErr = theano_train_entry(rudra::MLPparams::_batchSize, 
				    minibatchX.buf, rudra::MLPparams::_numInputDim, 
				    minibatchY.buf, rudra::MLPparams::_numClasses);
    //    return;
    //  }

    //    trainMBErr = nn->trainNetworkNoUpdate(minibatchX, minibatchY); // who will gc minibatchX and minibatchY ?
    //std::cout<<"here here trainMBErr "<<trainMBErr<<std::endl;

    // TODO: return the Rail of the updates
}

void serUpdates(float *updates) {
  //  if (THEANO_LEARNER==0) {
    theano_get_update_entry(updates);
    return;
    //  }
    //  nn->serializeUpdates(updates);
}

float trainOneMBErr(){
    return trainMBErr; 
}

void deSerializeWeights(float *weights){
  //  if (THEANO_LEARNER==0) {
    theano_set_param_entry(weights);
    //    return;
    //  }
    //    nn->deserialize(weights);
}
/**
 * update learning rate
 */
void updateLearningRate(){
  //  if (THEANO_LEARNER==0) {
    // vj TODO: Get Theano to understand about updating learning rates.
    return;
    //  }
    //    nn->setLearningRate(rudra::MLPparams::_alphaDecay);
}

/** vj TODO: Need to update the parameter server so that it can update the weights
  without requiring a NN be created. In essence the update rule is independent of 
  the network structure.

  There is no reason for this code to be in C++ -- until we are ready to use BLAS calls. 
 */

//////////////////// methods on the side of param server ///////////////
void syncSum(float *delta, float multiplier){
    // this delta is from the outside Rail
    nn->deserializeUpdates(delta);
    for(int i = 0; i < nn->N; ++i){           
	nn->L[i]->sumUpdate(multiplier);
    }
}

void syncUpdate(){
    for(int i = 0; i < nn->N; ++i){           
	nn->L[i]->applyUpdate();
    }
}
void asyncUpdate(float *delta, float multiplier){
    nn->deserializeUpdates(delta);
    // s1:
    for(int i = 0; i < nn->N; ++i){              
	//nn->L[i]->sumUpdate(1.0f);
	nn->L[i]->sumUpdate(multiplier);
    }
    //s2:
    for(int _i = 0; _i < nn->N; ++_i){
	nn->L[_i]->applyUpdate();                                          
    }
    
}

void serializeWeights(float *weights){
    nn->serialize(weights);
}


/**
 * assuming this call is always 
 */
long uuNum = 0;
#define UU_NOP_FLAG  1 // just plain sum update
#define UU_SMB_FLAG 2 // finish a "super mb" (i.e., super mini-batch size)
#define UU_EPOCH_FLAG 4 // finish the entire epoch (now the time to send a test to test server) 

int unifiedUpdate(float* delta, float multiplier){
    int result = 0;
    nn->deserializeUpdates(delta);
    for(int i = 0; i < nn->N; ++i){           
	nn->L[i]->sumUpdate(multiplier);
	
    }
    result |= UU_NOP_FLAG;
    uuNum++;
    if( (uuNum % NUM_LEARNER ) == 0){ // see a super mini-batch
	for(int _i = 0; _i < nn->N; ++_i){
	    nn->L[_i]->applyUpdate();                                          
	}
	result |= UU_SMB_FLAG;
    }
    if( (uuNum % NUM_MB_PER_EPOCH) == 0 ){ // see the entire epoch
	nn->setLearningRate(rudra::MLPparams::_alphaDecay);
	result |= UU_EPOCH_FLAG;
    }
    return result;
}

long getUUNum(){
    return uuNum;
}

////////////////// SS Section //////////////////////////
float testErr = -1.0f;

float testOneEpoch(float *weights){
    
    rudra::Network *nn = rudra::Network::readFromConfigFile(rudra::MLPparams::MLPCfg["layerCfgFile"]);
      nn->deserialize(weights);
      std::cout<<"[in native land] before testing network, network size: "<<nn->networkSize<<std::endl;
      testErr = nn->testNetwork(testData, testLabels);
      std::cout<<"[in naitve land] after testing network, testErr: "<<testErr<<std::endl;
      delete nn;
      return testErr;
}

float testOneEpoch(){
    testErr = nn->testNetwork(testData, testLabels);
    return testErr;
}
int loadTestData(){
    // 	std::cout<<"native land load test data "<<std::endl;
    // testData        = rudra::readBinMat(rudra::MLPparams::_testData);
    // testLabels      = rudra::readBinMat(rudra::MLPparams::_testLabels);
    // std::cout<<"naitve land after readBinMats"<<std::endl;
    // testData.transposeInPlace(); // bug fix, on May 5, 2015, don't forget to transpose testData!
    // testLabels.transposeInPlace(); // bug fix, on May 5, 2015, don't forget to transpose testLabels!
    std::cout<<"native land after transposition"<<std::endl;
    return 0;
}

rudra::GPFSSampleClient *testSC = NULL;
void initTestSC(){
	char agentName[21];
	sprintf(agentName, "LearnerAgent %6d", pid);
	rudra::BinarySampleSeqReader *binReader =
				new rudra::BinarySampleSeqReader(rudra::MLPparams::_testData, rudra::MLPparams::_testLabels, rudra::MLPparams::_numTestSamples);
	testSC = new rudra::GPFSSampleClient(agentName, true, binReader);

}
float testOneEpochSC(float *weights){
	assert(testSC != NULL );
	size_t batchSize = std::min(rudra::MLPparams::_numTestSamples,
			rudra::MLPparams::_batchSize);
	size_t minibatchSize = rudra::MLPparams::_numTestSamples / batchSize;
	size_t numMB = std::max((size_t) 1, minibatchSize);
	rudra::MatrixContainer<float> testX, testY;
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

////////////////// end of SS Section //////////////////
