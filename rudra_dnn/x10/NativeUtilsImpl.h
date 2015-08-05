#ifndef __NATIVE_UTILS_IMPL_H
#define __NATIVE_UTILS_IMPL_H
#include "x10rt.h"
#define X10_LANG_FLOAT_H_NODEPS
#include <x10/lang/Float.h>
#undef X10_LANG_FLOAT_H_NODEPS
#define X10_LANG_FLOAT_H_NODEPS
#include <x10/lang/Float.h>
#undef X10_LANG_FLOAT_H_NODEPS
#include<x10/lang/Rail.h>


/***  libcnn related code ***/
//#include<MLPparams.h>

/*** end of licnn related code **/
// convert a rail of float to a ptr
//int rail2Ptr(::x10::lang::rail<x10_float>* r, const char* ptrName);


/// utility methods 
::x10::lang::Rail<x10_float>* ptr2Rail();
::x10::lang::Rail<x10_float>* ptr2Rail(float *src, size_t size);


//// methods shared by Param Server and Learner Agents
void initNativeLand(long id, const char* confName, long numLearner);
void initAsLA();
void initAsSS();
void initAsPS();
void initXY(); // added on May 1, 2015, X: minibatch training data, Y: minibatch label
void initTrainSC(); // added on May 1, 2015: init gpfs sample client
int initNetwork();
int getNetworkSize();

//// methods on the learner agents
int loadTrainData();
int loadTestData();
/**
 * @return the parameter delta
 */
void selTrainMB();
void trainOneMB();
void serUpdates(float *updates);
void ensureLoaded(const  float *p);
float trainOneMBErr();


float testOneEpoch(float* weights); // added on May 5, 2015

/**
 * @return test error after one epoch
 */
float testOneEpoch();

void deSerializeWeights(float *weights);

void updateLearningRate();

////// methods on the param server
void syncSum(float* delta, float multiplier);
void syncUpdate();
void asyncUpdate(float *delta, float multiplier);
void serializeWeights(float *weights);

// will be used by both hardsync and smart protocol, added on June 7, 2015
int unifiedUpdate(float* delta, float multiplier);

// return the unified update number
long getUUNum();


///////////////////// SS Section /////////////////////////////////

void initTestSC();
float testOneEpochSC(float *weights);
//////////////////// end of SS Section //////////////////////////
#endif
