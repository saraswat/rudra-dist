#ifndef __NATIVE_LEARNER_IMPL_H
#define __NATIVE_LEARNER_IMPL_H
#include "x10rt.h"
#define X10_LANG_FLOAT_H_NODEPS
#include <x10/lang/Float.h>
#undef X10_LANG_FLOAT_H_NODEPS
#define X10_LANG_FLOAT_H_NODEPS
#include <x10/lang/Float.h>
#undef X10_LANG_FLOAT_H_NODEPS
#include <x10/lang/Rail.h>

#include <rudra/MLPparams.h>
// #include <rudra/Network.h>
#include <rudra/util/MatrixContainer.h>
#include <rudra/io/GPFSSampleClient.h> // x10-impl always assumes GPFSSampleClient

#include <cstdlib>
#include <cstring>
#include <sys/time.h>

namespace xrudra {
  class NativeLearner {

  public:
    NativeLearner();
    NativeLearner* _make();

    long pid;    
    // Only for Native Rudra learner
    //    rudra::Network* nn;
    float trainMBErr;
    rudra::MatrixContainer<float> minibatchX;    
    rudra::MatrixContainer<float> minibatchY;    
    rudra::MatrixContainer<float> Data;    
    rudra::MatrixContainer<float> Labels;    
    rudra::MatrixContainer<float> testData;    
    rudra::MatrixContainer<float> testLabels;    
    rudra::GPFSSampleClient* trainSC;    
    rudra::GPFSSampleClient* testSC;
    int NUM_LEARNER;
    int NUM_MB_PER_EPOCH;

    void *learner_data;
    int (*leaner_init)(void **data, struct param[], size_t numParams);
    void (*learner_destroy)(void *data);
    size_t (*leaner_netsize)(void *data);
    float (*leaner_train)(void *data, uint32 batchSize,
                          const float *features, ssize_t numInputDims,
                          const float *targets, ssize_t numClasses);
    float (*leaner_test)(void *data, uint32 batchSize,
                         const float *features, ssize_t numInputDims,
                         const float *targets, ssize_t numClasses);
    void (*learner_getgrads)(void *data, float *updates);
    void (*learner_accgrads)(void *data, float *updates);
    void (*leaner_updatelr)(void *data, float newLR);
    void (*leaner_getweights)(void *data, float *weights);
    void (*leaner_setweights)(void *data, float *weights);
    void (*leaner_updweights)(void *data, float *weights, size_t numMB);

    std::string cfgFile;
    float testErr;

    //// methods shared by Param Server and Learner Agents
    void initNativeLand(long id, const char* confName, long numLearner);
    void initAsLA(bool isReconciler);
    void initXY(); // added on May 1, 2015, X: minibatch training data, Y: minibatch label
    void initTrainSC(); // added on May 1, 2015: init gpfs sample client
    int initNetwork(bool isReconciler);
    void initPSU(std::string _solverType);
    void setMeanFile(std::string _fileName);
    int getNetworkSize();

    /**
     * @return the parameter delta
     */
    void loadMiniBatch();
    float trainMiniBatch();
    void  getGradients(float *updates);
    void  accumulateGradients(float *updates);
    float trainOneMBErr();

    float testOneEpoch(float* weights); // added on May 5, 2015

    /**
     * @return test error after one epoch
     */
    float testOneEpoch();

    void serializeWeights(float *weights);
    void deserializeWeights(float *weights);
  
    void updateLearningRate(long curEpochNum);

    ////// methods on the param server
    void syncSum(float* delta, float multiplier);
    void syncUpdate();
    void asyncUpdate(float *delta, float multiplier);

    // will be used by both hardsync and smart protocol, added on June 7, 2015
    int unifiedUpdate(float* delta, float multiplier);
    void acceptGradients(float *grad, size_t numMB);

    ///////////////////// SS Section /////////////////////////////////

    void initTestSC();
    void initTestSC(long id, size_t numLearner);
    float testOneEpochSC(float *weights);
    long getTestNum();
    //////////////////// end of SS Section //////////////////////////

  };
}
#endif
