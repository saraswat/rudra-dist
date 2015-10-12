/*
 * MLPparams.h
 *
 *  Created on: Dec 7, 2014
 *      Author: suyog
 */

#ifndef RUDRA_MLPPARAMS_H_
#define RUDRA_MLPPARAMS_H_

#include "rudra/util/defs.h"
#include <sstream>
#include <map>
#include <vector>
#include <sys/stat.h>
#include <sys/time.h>
#include <endian.h>
const time_t ctt = time(NULL);

#define RUDRA_TIME asctime(localtime(&ctt))

typedef std::map<std::string, std::string> keyMap;

namespace rudra {
template<typename T> std::string toString(T tmp) {
        std::ostringstream out;
        out << tmp;
        return out.str();
}

class MLPparams {
public:


        /*MLP parameters that are being read from .cfg file*/
        //------------------------------------- 
        static uint32            _batchSize;// mini-batch size in stochastic gradient descent
        static uint32            _numTrainSamples;    // number of training samples
        static uint32            _numInputDim;        // input dimension
        static uint32            _numClasses;    // number of output classes
        static uint32            _numTestSamples;    // number of test samples
        static uint32            _numEpochs;    // total number of training epochs
        static uint32            _testInterval; // testing interval. set to 0 for no testing

        static std::string      _trainData;    // file name containing the training data
        static std::string      _trainLabels; // file name containing the training labels
        static std::string      _testData;    // file name containing the test data
        static std::string      _testLabels;    // file name containing the test labels

        struct LearningRateMultiplier{
                static std::vector<float>   _lr;
                static std::string          _schedule;    //rate schedule : const/step/power/exp/custom             
                static std::string          _lrFile;              //if schedule = custom, lrFile will contain the learning rate for every epoch
                static std::vector<int>     _epochs;
                static float                  _beta;
                static float                 _gamma;
        };
        

        /*MLP parameters that are being read from command line*/
        static float          _adaDeltaRho; 
        static float          _adaDeltaEpsilon; 
        static std::string    _givenFileName; // -f (FIXME: split into cfg and network files)
        static std::string    _restartFileName;// -r
        static std::string    _jobID;          // -j
        static int            _randSeed;       // -s (to initialize random number generator)
        static int            _sampleClient;   // -sc
        static float          _lrMult;         // -mul
        static std::string    _meanFile;       // -meanFile
        static std::string    _solver;         // type of solver adagrad or sgd
        static float          _mom;            // momentum
        static int            _printInterval;  // how often to print on the console?
        static std::string    _allowedGPU;     // -gpu 0,1,3
        /*Other paramaters      */

        static uint32        _chkptInterval;  // checkpoint interval
        static std::string    _resFileName;   // name of the final result file
    static std::string           _rudraHome;  // directory containing this code.
    static std::string              _logDir;  // working directory
        
        static keyMap                MLPCfg;    // stores MLP config
        static uint32                _epoch;    // stores the current epoch number
        static uint32                   _mb;    // stores the current minibatch number
        static bool            _isInference;    // Set to true if the network is being used to perform testing/inference, rather than training. Default value is false

        static const std::string paramList[];    // stores the list of MLP parameters
        static const int            paramNum;    // number of MLP parameters

        //---------------------------------------
        /* static methods */
    static void     initMLPparams(std::string S);

        static void             setMLPdefaults();
        static bool     setParam(keyMap inp, std::string param);
        static void             setWD();
        static void     setRudraHome();
        static void     setJobID(std::string);
        static void     setChkptInterval(int);
        static void     setLearningRateMultiplierSchedule();

        static void     readBinHeader(std::string, int& r, int& c);

};

} /* namespace rudra */
#endif /* RUDRA_MLPPARAMS_H_ */
