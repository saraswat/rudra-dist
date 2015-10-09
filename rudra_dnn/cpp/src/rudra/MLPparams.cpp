#include "rudra/MLPparams.h"
#include "rudra/io/SampleClient.h"
#include "rudra/util/Logger.h"
#include "rudra/util/Parser.h"
#include <algorithm>
#include <iomanip>
#include <iostream>
#include <cmath>
#include <cstdlib>
#include <cstring>

namespace rudra {
const std::string MLPparams::paramList[] = { "trainData", "trainLabels",
		"testData", "testLabels", "layerCfgFile", "testInterval", "meanFile", "chkptInterval", "numTrainSamples", "numTestSamples", "numInputDim",
                "numClasses", "numEpochs", "batchSize", "learningSchedule", "gamma", "beta", "lrFile", "epochs" };

  const int MLPparams::paramNum = sizeof(MLPparams::paramList) / sizeof(*MLPparams::paramList);



  // set defaults
  uint32      MLPparams::_numTrainSamples = RUDRA_DEFAULT_INT;
  uint32      MLPparams::_numClasses      = RUDRA_DEFAULT_INT;
  uint32      MLPparams::_numInputDim     = RUDRA_DEFAULT_INT;
  uint32      MLPparams::_numTestSamples  = RUDRA_DEFAULT_INT;
  uint32      MLPparams::_numEpochs       = RUDRA_DEFAULT_INT;
  uint32      MLPparams::_batchSize       = RUDRA_DEFAULT_INT;
  uint32      MLPparams::_testInterval    = RUDRA_DEFAULT_INT;
  std::string MLPparams::_trainData       = RUDRA_DEFAULT_STRING;
  std::string MLPparams::_trainLabels     = RUDRA_DEFAULT_STRING;
  std::string MLPparams::_testData        = RUDRA_DEFAULT_STRING;
  std::string MLPparams::_testLabels      = RUDRA_DEFAULT_STRING;
  
  uint32      MLPparams::_chkptInterval   = RUDRA_DEFAULT_INT;
  std::string MLPparams::_logDir          = RUDRA_DEFAULT_STRING;
  std::string MLPparams::_rudraHome       = RUDRA_DEFAULT_STRING;
  std::string MLPparams::_resFileName     = RUDRA_DEFAULT_STRING;
  uint32      MLPparams::_epoch           = RUDRA_DEFAULT_INT;
  uint32      MLPparams::_mb              = RUDRA_DEFAULT_INT;

  // learning rate related parameters
  std::string MLPparams::LearningRateMultiplier::_lrFile = RUDRA_DEFAULT_STRING;
  float		  MLPparams::LearningRateMultiplier::_beta   = 1.0f;
  float		  MLPparams::LearningRateMultiplier::_gamma  = 1.0f;
  std::string MLPparams::LearningRateMultiplier::_schedule = "constant";
  std::vector<int> 	 MLPparams::LearningRateMultiplier::_epochs;
  std::vector<float> MLPparams::LearningRateMultiplier::_lr;

  bool        MLPparams::_isInference     = false;

  // Parameters from command line are initialized to defaults to take effect
  //  if no value provided
  float        MLPparams::_adaDeltaRho     = 0.95;
    float        MLPparams::_adaDeltaEpsilon = 1e-6;
  std::string  MLPparams::_givenFileName   = RUDRA_DEFAULT_STRING;   // -f (FIXME: split into cfg and network files)
  std::string  MLPparams::_restartFileName = RUDRA_DEFAULT_STRING;   // -r
  std::string  MLPparams::_jobID           = RUDRA_DEFAULT_STRING;   // -j 
  int          MLPparams::_randSeed        = 12345;      			 // -s (to initialize random number generator)
  int          MLPparams::_sampleClient    = 3;            		     // -sc
  float        MLPparams::_lrMult          = 1.0;                    // -mul
  std::string  MLPparams::_allowedGPU	   = RUDRA_DEFAULT_STRING;   // -gpu     default if no list given
  std::string  MLPparams::_meanFile		   = RUDRA_DEFAULT_STRING;   // -meanFile
  std::string  MLPparams::_solver		   = RUDRA_DEFAULT_STRING;	 // type of solved: adagrad or sgd
  float		   MLPparams::_mom			   = RUDRA_DEFAULT_INT;	 	 // momentum. overrides values in .cnn
  int		   MLPparams::_printInterval   = 1;						 // -printInterval : how often to print to the interval

  keyMap MLPparams::MLPCfg;		// stores MLP config

void MLPparams::setMLPdefaults() {

	MLPparams::MLPCfg["trainData"] 		= MLPparams::_trainData;
	MLPparams::MLPCfg["trainLabels"]	= MLPparams::_trainLabels;
	MLPparams::MLPCfg["testData"] 		= MLPparams::_testData;
	MLPparams::MLPCfg["testLabels"] 	= MLPparams::_testLabels;
	MLPparams::MLPCfg["meanFile"] 	 	= MLPparams::_meanFile;
	MLPparams::MLPCfg["layerCfgFile"] 	= MLPparams::_rudraHome + "examples/mnist.cnn";
	MLPparams::MLPCfg["testInterval"]	= "1";
	MLPparams::MLPCfg["chkptInterval"]	= "10";
	MLPparams::MLPCfg["numTrainSamples"]= "60000";
	MLPparams::MLPCfg["numTestSamples"] = "10000";
	MLPparams::MLPCfg["numInputDim"]    = "784";
	MLPparams::MLPCfg["numClasses"]		= "10";
	MLPparams::MLPCfg["numEpochs"]		= "100";
	MLPparams::MLPCfg["batchSize"] 		= "100";


	MLPparams::MLPCfg["learningSchedule"] = "constant";
	MLPparams::MLPCfg["gamma"]			= "1";
	MLPparams::MLPCfg["beta"]			= "1";
	MLPparams::MLPCfg["lrFile"]			= MLPparams::LearningRateMultiplier::_lrFile;
	MLPparams::MLPCfg["epochs"]			= "0";

}
bool MLPparams::setParam(keyMap inp, std::string param) {
	std::stringstream s;

	bool retval = false;
	if (inp.find(param) != inp.end()) {
		MLPparams::MLPCfg[param] = inp[param];
		s << "\tMLPparams::" << std::setw(15) << param << " "
				<< std::setfill('.') << std::setw(10) << " "
				<< MLPparams::MLPCfg[param];
		retval = true;
	}else {
                s << "\tMLPparams::" << std::setw(15) << param << " "
                                << std::setfill('.') << std::setw(10) << " " << MLPparams::MLPCfg[param]
                                << " (default)";
        }

	rudra::Logger::logInfo(s.str());
	return retval;
}

void MLPparams::setRudraHome(){
		char * rudra_home = std::getenv("RUDRA_HOME");
		if(!rudra_home){
				rudra::Logger::logFatal("RUDRA_HOME environment variable not set. Can not proceed!");
		}else{
				MLPparams::_rudraHome = std::string(rudra_home);
		}

		if(MLPparams::_rudraHome.at(MLPparams::_rudraHome.length() - 1) != '/' ){
				MLPparams::_rudraHome.append("/");
		}


}

void MLPparams::setLearningRateMultiplierSchedule(){

	std::string schedule = MLPparams::LearningRateMultiplier::_schedule;
	float gamma 		 = MLPparams::LearningRateMultiplier::_gamma;
	float beta 			 = MLPparams::LearningRateMultiplier::_beta;

	if(std::strcmp(schedule.c_str(),"constant") == 0){
		// constant multiplier
		for(int i = 0; i < MLPparams::_numEpochs; ++i){
			MLPparams::LearningRateMultiplier::_lr.push_back(1.0f);
		}

	}else if(std::strcmp(schedule.c_str(),"exponential") == 0){
		// exponential multiplier
		// m(i) = gamma^i;
		MLPparams::LearningRateMultiplier::_lr.push_back(1.0f);
		for(int i = 1; i < MLPparams::_numEpochs; ++i){
			MLPparams::LearningRateMultiplier::_lr.push_back(MLPparams::LearningRateMultiplier::_lr[i-1]
			                                                 *gamma);
		}
	}else if(std::strcmp(schedule.c_str(),"power")== 0){
			// exponential multiplier
			// m(i) = 1/(1+i*gamma)^beta;

		for(int i = 0; i < MLPparams::_numEpochs; ++i){
			MLPparams::LearningRateMultiplier::_lr.push_back(1.0f/powf((1+i*gamma),beta));
		}

	}else if(std::strcmp(schedule.c_str(),"step")== 0){
				// piecewise constant

		float mul = 1.0f;
		std::vector<int> epochs = MLPparams::LearningRateMultiplier::_epochs;
		for(int i = 0; i < MLPparams::_numEpochs; ++i){

			if(std::find(epochs.begin(), epochs.end(), i) != epochs.end())
				mul *= gamma;

			MLPparams::LearningRateMultiplier::_lr.push_back(mul);
		}


	}else{
		// default: constant
		for(int i = 0; i < MLPparams::_numEpochs; ++i){
			MLPparams::LearningRateMultiplier::_lr.push_back(1.0f);
		}
	}



}

void MLPparams::initMLPparams(std::string S) {

    MLPparams::setRudraHome();
	MLPparams::setMLPdefaults();
	rudra::Parser p(S);
	p.parseFile();

	for (int i = 0; i < MLPparams::paramNum; i++) {
		MLPparams::setParam(p._params[0], MLPparams::paramList[i]);
	}


	MLPparams::_trainData 	= MLPparams::MLPCfg["trainData"];
	MLPparams::_trainLabels = MLPparams::MLPCfg["trainLabels"];
	MLPparams::_testData 	= MLPparams::MLPCfg["testData"];
	MLPparams::_testLabels 	= MLPparams::MLPCfg["testLabels"];
	MLPparams::_meanFile 	= MLPparams::MLPCfg["meanFile"];

	MLPparams::_testInterval = convert::string_to_T<uint32>(
			MLPparams::MLPCfg["testInterval"]);
	MLPparams::_chkptInterval = convert::string_to_T<uint32>(
			MLPparams::MLPCfg["chkptInterval"]);
	MLPparams::_numTrainSamples = convert::string_to_T<uint32>(
			MLPparams::MLPCfg["numTrainSamples"]);
	MLPparams::_numTestSamples = convert::string_to_T<uint32>(
			MLPparams::MLPCfg["numTestSamples"]);
	MLPparams::_numInputDim = convert::string_to_T<uint32>(
			MLPparams::MLPCfg["numInputDim"]);
	MLPparams::_numClasses = convert::string_to_T<uint32>(
			MLPparams::MLPCfg["numClasses"]);
	MLPparams::_numEpochs = convert::string_to_T<uint32>(
			MLPparams::MLPCfg["numEpochs"]);
	MLPparams::_batchSize = convert::string_to_T<uint32>(
			MLPparams::MLPCfg["batchSize"]);

	MLPparams::LearningRateMultiplier::_schedule = MLPparams::MLPCfg["learningSchedule"];

	MLPparams::LearningRateMultiplier::_gamma = convert::string_to_T<float>(
				MLPparams::MLPCfg["gamma"]);

	MLPparams::LearningRateMultiplier::_beta = convert::string_to_T<float>(
				MLPparams::MLPCfg["beta"]);

	MLPparams::LearningRateMultiplier::_lrFile = MLPparams::MLPCfg["lrFile"];

	std::stringstream ss;
	std::string temp;
	ss << MLPparams::MLPCfg["epochs"];
	while(!ss.eof()){
		getline(ss,temp,',');
		MLPparams::LearningRateMultiplier::_epochs.push_back(convert::string_to_T<int>(temp));
	}

	//sanity check
	if (MLPparams::_numTrainSamples == 0)
		rudra::Logger::logFatal(
				"MLPparams::initMLPparams()::invalid value for numTrainSamples");

	if (MLPparams::_numTestSamples == 0)
		rudra::Logger::logFatal(
				"MLPparams::initMLPparams()::invalid value for numTestSamples");

	if (MLPparams::_numInputDim == 0)
		rudra::Logger::logFatal(
				"MLPparams::initMLPparams()::invalid value for numInputDim");

	if (MLPparams::_numClasses == 0)
		rudra::Logger::logFatal(
				"MLPparams::initMLPparams()::invalid value for numClasses");

	if (MLPparams::_batchSize == 0)
		rudra::Logger::logFatal(
				"MLPparams::initMLPparams()::invalid value for batchSize");

	if (MLPparams::_numEpochs == 0)
		rudra::Logger::logFatal(
				"MLPparams::initMLPparams()::invalid value for numEpochs");
	if (MLPparams::_testInterval > MLPparams::_numEpochs){
		MLPparams::_testInterval = MLPparams::_numEpochs;
		rudra::Logger::logInfo(
				"MLPparams::initMLPparams()::setting testInterval = numEpochs");
	}
	
	MLPparams::setLearningRateMultiplierSchedule();

}

void MLPparams::setJobID(std::string s){
	
	MLPparams::_jobID = s;

}

void MLPparams::setChkptInterval(int i){
	if (i < 0 || i > MLPparams::_numEpochs){
	
		MLPparams::_chkptInterval = MLPparams::_numEpochs;
	
	}else{
		
		MLPparams::_chkptInterval = i;

	}		
}
void MLPparams::setWD() {

        struct stat st;
        uint16 cnt = 0;
        std::string dirName, fileName;
        int retval;

        dirName = MLPparams::_rudraHome + "LOG/" + MLPparams::_jobID + "/";

        stat(dirName.c_str(),&st);
        if (stat(dirName.c_str(), &st) == -1){

        	rudra::Logger::logInfo("MLPparams::setWD: Creating working directory in " + dirName + "  ...");
        	std::string command = "mkdir -p " + dirName;
        	retval = system(command.c_str()); // system call to execute the command

	}else{
		rudra::Logger::logFatal("MLPparams::setWD: Directory " + dirName + " already exists. Launch again with new job_id");
	}
        
	MLPparams::_logDir = dirName;
	MLPparams::_resFileName = MLPparams::_logDir + MLPparams::_jobID + ".OUT";
}

void MLPparams::readBinHeader(std::string fileName, int& r, int& c){

	std::ifstream f1(fileName.c_str(), std::ios::in | std::ios::binary); //open file for reading in binary mode
	if(!f1){
		std::cout << "MLPparams::readBinHeader::Error! failed to open file: " << fileName << std::endl;
		exit(EXIT_FAILURE);
	}

	int r1,c1;

	f1.read((char*)&r1,sizeof(uint32));	// read number of rows
	f1.read((char*)&c1,sizeof(uint32));	// read number of cols

#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
        // swap byte order
        r1 = be32toh(r1);
        c1 = be32toh(c1);
#endif

	if (r1 < 0 || c1 < 0){
		std::cout << "MLPparams::readBinHeader::Invalid matrix dimensions::" << r1 << " | " << c1 << " in file: " << fileName <<std::endl;
		exit(EXIT_FAILURE);
	}

	r = r1;
	c = c1;

}

} /* namespace rudra */

