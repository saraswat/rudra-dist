#include <rudra/NativeLearner.h>
#include <iostream>

namespace rudra {

NativeLearner::NativeLearner(long id) : pid(id) {
    std::cout << ">>> NativeLearner::NativeLearner(" << id << ")" << std::endl;
}

long NativeLearner::getNumEpochs() {
    std::cout << ">>> NativeLearner::getNumEpochs()" << std::endl;
    return 1;
}

long NativeLearner::getNumTrainingSamples() {
    std::cout << ">>> NativeLearner::getNumTrainingSamples()" << std::endl;
    return 2;
}

long NativeLearner::getMBSize() {
    std::cout << ">>> NativeLearner::getMBSize()" << std::endl;
    return 1;
}

void NativeLearner::setLoggingLevel(int level) {
    std::cout << ">>> NativeLearner::setLoggingLevel(" << level << ")" << std::endl;
}

void NativeLearner::setJobDir(std::string jobDir) {
    std::cout << ">>> NativeLearner::setJobDir(\"" << jobDir << "\")" << std::endl;
}

void NativeLearner::setAdaDeltaParams(float rho, float epsilon,
                                    float drho, float depsilon) {
    std::cout << ">>> NativeLearner::setAdaDeltaParams(" << rho << ", " << epsilon << ", " << drho << ", " << depsilon << ")" << std::endl;
}

void NativeLearner::setMeanFile(std::string fn) {
    std::cout << ">>> NativeLearner::setMeanFile(\"" << fn << "\")" << std::endl;
}

void NativeLearner::setSeed(long id, int seed, int defaultSeed) {
    std::cout << ">>> NativeLearner::setSeed(" << id << ", " << seed << ", " << defaultSeed << ")" << std::endl;
}

void NativeLearner::setMoM(float mom) {
    std::cout << ">>> NativeLearner::setMoM(" << mom << ")" << std::endl;
}

void NativeLearner::setLRMult(float mult) {
    std::cout << ">>> NativeLearner::setLRMult(" << mult << ")" << std::endl;
}

void NativeLearner::setWD() {
    std::cout << ">>> NativeLearner::setWD()" << std::endl;
}

void NativeLearner::initFromCFGFile(std::string confName) {
    std::cout << ">>> NativeLearner::initFromCFGFile(\"" << confName << "\")" << std::endl;
}

void NativeLearner::cleanup() {

}

void NativeLearner::checkpointIfNeeded(int whichEpoch) {
    std::cout << ">>> NativeLearner::checkpointIfNeeded(" << whichEpoch << ")" << std::endl;
}

void NativeLearner::initAsLearner(std::string weightsFile, std::string solverType) {
    std::cout << ">>> NativeLearner::initAsLearner(\"" << weightsFile << "\", \"" << solverType << "\")" << std::endl;
}

void NativeLearner::initAsTester(long placeID, std::string solverType) {
    std::cout << ">>> NativeLearner::initAsTester(" << placeID << ", \"" << solverType << "\")" << std::endl;
}

int NativeLearner::getNetworkSize() {
    return 1;
}

float NativeLearner::trainMiniBatch() {
    std::cout << ">>> NativeLearner::trainMiniBatch()" << std::endl;
}

void NativeLearner::getGradients(float *gradients) {
    std::cout << ">>> NativeLearner::getGradients(" << gradients << ")" << std::endl;
}

void NativeLearner::accumulateGradients(float *gradients) {
    std::cout << ">>> NativeLearner::accumulateGradients(" << gradients << ")" << std::endl;
}

void NativeLearner::updateLearningRate(long curEpochNum) {
    std::cout << ">>> NativeLearner::updateLearningRate(" << curEpochNum << ")" << std::endl;
}

void NativeLearner::serializeWeights(float *weights) {
    std::cout << ">>> NativeLearner::serializeWeights(" << weights << ")" << std::endl;
}

void NativeLearner::deserializeWeights(float *weights) {
    std::cout << ">>> NativeLearner::deserializeWeights(" << weights << ")" << std::endl;
}


void NativeLearner::acceptGradients(float *grad, size_t numMB) {
    std::cout << ">>> NativeLearner::acceptGradients(" << grad << ", " << numMB << ")" << std::endl;
}

float NativeLearner::testOneEpochSC(float *weights, size_t numTesters) {
    std::cout << ">>> NativeLearner::testOneEpochSC(" << weights << ", " << numTesters << ")" << std::endl;
return 100.0;
}

} // namespace rudra

