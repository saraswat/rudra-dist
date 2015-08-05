#include "rudra/MLPparams.h"
#include "rudra/io/SampleClient.h"
#include "rudra/util/Tracing.h"
#include <iomanip>
#include <iostream>
#include <cstdlib>
namespace rudra {
const std::string MLPparams::paramList[] = { "trainData", "trainLabels",
		"testData", "testLabels", "layerCfgFile", "testInterval", "meanFile", "chkptInterval", "numTrainSamples", "numTestSamples", "numInputDim",
                "numClasses", "numEpochs", "batchSize", "alphaDecay", "platform" };

  const int MLPparams::paramNum = sizeof(MLPparams::paramList) / sizeof(*MLPparams::paramList);

  // set defaults
  uint32      MLPparams::_numTrainSamples = RUDRA_DEFAULT_INT;
  uint32      MLPparams::_numClasses      = RUDRA_DEFAULT_INT;
  uint32      MLPparams::_numInputDim     = RUDRA_DEFAULT_INT;
  uint32      MLPparams::_numTestSamples  = RUDRA_DEFAULT_INT;
  uint32      MLPparams::_numEpochs       = RUDRA_DEFAULT_INT;
  uint32      MLPparams::_batchSize       = RUDRA_DEFAULT_INT;
  uint32      MLPparams::_testInterval    = RUDRA_DEFAULT_INT;
  float       MLPparams::_alphaDecay      = RUDRA_DEFAULT_FLOAT;
  std::string MLPparams::_trainData       = RUDRA_DEFAULT_STRING;
  std::string MLPparams::_trainLabels     = RUDRA_DEFAULT_STRING;
  std::string MLPparams::_testData        = RUDRA_DEFAULT_STRING;
  std::string MLPparams::_testLabels      = RUDRA_DEFAULT_STRING;
  
  uint32      MLPparams::_chkptInterval   = RUDRA_DEFAULT_INT;
  uint32      MLPparams::_labelDim        = RUDRA_DEFAULT_INT;
  std::string MLPparams::_logDir          = RUDRA_DEFAULT_STRING;
  std::string MLPparams::_rudraHome       = RUDRA_DEFAULT_STRING;
  std::string MLPparams::_resFileName     = RUDRA_DEFAULT_STRING;
  uint32      MLPparams::_epoch           = RUDRA_DEFAULT_INT;
  uint32      MLPparams::_mb              = RUDRA_DEFAULT_INT;

  bool        MLPparams::_isInference     = false;

  // Parameters from command line are initialized to defaults to take effect if no value provided
  std::string  MLPparams::_givenFileName   = RUDRA_DEFAULT_STRING;   // -f (FIXME: split into cfg and network files)
  std::string  MLPparams::_restartFileName = RUDRA_DEFAULT_STRING;   // -r
  std::string  MLPparams::_jobID           = RUDRA_DEFAULT_STRING;   // -j 
  int          MLPparams::_randSeed        = 12345;      	     // -s (to initialize random number generator)
  int          MLPparams::_sampleClient    = SC_NULL;                // -sc

  float        MLPparams::_lrMult          = 1.0;                    // -mul
  std::string  MLPparams::_meanFile	   = RUDRA_DEFAULT_STRING;   // -meanFile
  std::string  MLPparams::_solver	   = RUDRA_DEFAULT_STRING;   // type of solved: adagrad or sgd
  float	       MLPparams::_mom	           = RUDRA_DEFAULT_INT;	     // momentum. overrides values in .cnn
  int	       MLPparams::_printInterval   = 1;			     // -printInterval : how often to print to the interval

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
	MLPparams::MLPCfg["numTrainSamples"]	= "60000";
	MLPparams::MLPCfg["numTestSamples"]     = "10000";
	MLPparams::MLPCfg["numInputDim"]        = "784";
	MLPparams::MLPCfg["numClasses"]		= "10";
	MLPparams::MLPCfg["numEpochs"]		= "100";
	MLPparams::MLPCfg["batchSize"] 		= "100";
	MLPparams::MLPCfg["alphaDecay"]		= "0.95";       


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
	MLPparams::_alphaDecay = convert::string_to_T<float>(
			MLPparams::MLPCfg["alphaDecay"]);


        // If platform not specified then _platform retains initial value
	//        try {MLPparams::_platform = strToPlatform(MLPparams::MLPCfg.at("platform"));}
	//        catch (...) {}

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

        	rudra::Logger::logInfo("MLPparams::setWD::Creating working directory in " + dirName + "  ...");
        	std::string command = "mkdir -p " + dirName;
        	retval = system(command.c_str()); // system call to execute the command

	}else{
		rudra::Logger::logFatal("MLPparams::setWD::" + dirName + " already exists. Launch again with new job_id");
	}
        
	MLPparams::_logDir = dirName;
	MLPparams::_resFileName = MLPparams::_logDir + MLPparams::_jobID + ".OUT";
}

void MLPparams::setLabelDim(int i){

//	if (i > MLPparams::_numClasses){
//		rudra::Logger::logFatal("MLPparams::setLabelDim:: Label dim: " +convert::T_to_string(i) + " can not be more than numClasses: " + convert::T_to_string(MLPparams::_numClasses));
//	}
	MLPparams::_labelDim = i;
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


  // Return argv[i++]
  // In addition, check for errors
  static std::string nextArg(int   &i,
                             int    argc, 
                             char*  argv[],
                             int    whichVal) // flag? val1? val2?
    throw (Exception)
  {
    switch (whichVal) {
    case 0: 
      RUDRA_CHECK(     i < argc, "Failed to test argc before getting next flag");
      break;
    case 1:
      RUDRA_CHECK_USER(i < argc, "Missing value for option '"        << argv[i-1] << "'");
      break;
    case 2:
      RUDRA_CHECK_USER(i < argc, "Missing second value for option '" << argv[i-2] << "'");
      break;
    default: 
      RUDRA_CHECK(false, "");
    }

    switch (whichVal) {
    case 0: 
      RUDRA_CHECK_USER(*argv[i] == '-', "Expected a flag starting with '-', not " << argv[i]);
      break;
    default:
      RUDRA_CHECK_USER(*argv[i] != '-', "Flag '" << argv[i] << "' where value expected");
    }
    
    return std::string(argv[i++]);  // real work
  }


  // Parse command line given as argv, and assign results to MLPparams.
  // Return true is successful, false if any errors
  // It is intended that any main() function calls this utility, 
  // which just does the parsing and error checking,
  // and the caller from main() then checks if it got what it needs.

  bool MLPparams::parseCommandLine(int   argc,
                                   char *argv[])
  {
    std::string flag, val1;   // user gives:     -flag value1 [value2]

    try { // parsing failures print message and throw exception
    
      for (int i = 1; i < argc; ) {
        flag = nextArg(i, argc, argv, 0);
        val1 = nextArg(i, argc, argv, 1);  // every flag needs

        if      (flag == "-f")   		MLPparams::_givenFileName   =                             val1;
        else if (flag == "-r")   		MLPparams::_restartFileName =                             val1;
        else if (flag == "-j")   		MLPparams::_jobID           =                             val1;
        else if (flag == "-s")   		MLPparams::_randSeed        = convert::string_to_T<int>(  val1);
        else if (flag == "-sc")  		MLPparams::_sampleClient    = convert::string_to_T<int>(  val1);
        else if (flag == "-mul") 		MLPparams::_lrMult          = convert::string_to_T<float>(val1);
        else if (flag == "-meanFile") 	MLPparams::_meanFile		= val1;
        else if (flag == "-solver") 	MLPparams::_solver 			= val1;
        else if (flag == "-mom")		MLPparams::_mom				= convert::string_to_T<float>(val1);
        else if (flag == "-printInterval") MLPparams::_printInterval= convert::string_to_T<int>	 (val1);
        else if (flag == "-t")   { // e.g. -t timing 2
          int val2 = convert::string_to_T<int>(nextArg(i, argc, argv, 2));
          Tracing::set(val1, val2);  
        }

        else RUDRA_CHECK_USER(false, 
                              "Invalid option -- '" << flag << "'");
      }
    } catch (Exception) {
      // If parsing failed, message has been printed.
      // Return false, allowing caller to handle the failure.
      // This does not quite work because convert::string_to_T()
      // exists rather than throw an exception.
      return false;
    }
    return true;
  }


/*
void MLPparams::writeParams() {
	if (_resFile) {
		_resFile << "\n";
	NN.writeParamsH5(MLPparams::MLPCfg["logDir"]+"test.h5");
	NN.writeParamsH5(MLPparams::MLPCfg["logDir"]+"test.h5");
		_resFile << "########################################################"
				<< std::endl;
		_resFile << "# MLP params:" << std::endl;
		_resFile << "# alpha decay   = " << MLPparams::_alphaDecay << std::endl;
		_resFile << "# batchsize     = " << MLPparams::_batchSize << std::endl;

		_resFile << "########################################################"
				<< std::endl;
	} else {
		rudra::Logger::logFatal(
				"MLPparams::writeParams()::Trouble writing to the output file...");
	}
}
*/
} /* namespace rudra */

