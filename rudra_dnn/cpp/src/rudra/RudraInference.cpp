/*
 * RudraInference.cpp
 *
 *  Created on: May 18, 2015
 *      Author: suyog
 */

// perform feedforward pass on a trained network

#include "rudra/Network.h"
#include "rudra/MLPparams.h"
#include "rudra/layers/Layer.h"
#include "rudra/math/Matrix.h"
#include "rudra/util/Logger.h"
#include "rudra/util/PerfTimer.h"
// include for sample client
#include "rudra/io/SampleClient.h"
#include "rudra/io/BinarySampleSeqReader.h"
#include "rudra/io/GPFSSampleClient.h"
#include "rudra/io/UnifiedBinarySampleSeqReader.h"
#include "H5Cpp.h"
#include <iostream>
#include <sys/time.h>

using namespace rudra;


void printUsage(std::string execName) {
	std::cerr << "Usage: " << execName << std::endl
			<< "Required:" << std::endl
			<< " \t-f /path/to/network_configuration_file/ " << std::endl
			<< " \t-m /path/to/model_parameter_file " << std::endl
			<< " \t-d /path/to/testDataFile/"<< std::endl
			<< "Optional: " << std::endl
			<< " \t-j job_id (default : rudra_inference)" << std::endl
			<< " \t-o layerName1,layerName2,..,layerNameN" << std::endl
			<< " \t-l /path/to/testLabelFile/" << std::endl
			<< " \t-meanFile /path/to/meanFile/"<< std::endl
			<< " \t-errorType top-1|top-5" << std::endl;
}


/**
 * set up global parameters
 */
std::string networkCfgFile = RUDRA_DEFAULT_STRING;
std::string jobID = "rudra_inference";
std::string modelFile =  RUDRA_DEFAULT_STRING;
std::vector<std::string> outputs;
std::string testDataFile  = RUDRA_DEFAULT_STRING;
std::string testLabelFile = RUDRA_DEFAULT_STRING;
std::string meanFileName  = RUDRA_DEFAULT_STRING;
std::string errorType	  = RUDRA_DEFAULT_STRING;

std::string outputFileName = RUDRA_DEFAULT_STRING;
bool writeOutputs = false;
bool doTest		  = false;
hsize_t maxdims[2] = {H5S_UNLIMITED, H5S_UNLIMITED};

void parseCmdLine(int argc, char **argv) {
	if (argc < 2) {
		printUsage(argv[0]);
		exit(1);
	}

	for (int i = 1; i < argc; ++i) {

		/* 1. network config file */
		if (std::string(argv[i]) == "-f") {
			if (i + 1 < argc) { // Make sure we aren't at the end of argv!
				networkCfgFile = argv[i + 1]; // Increment 'i' so we don't get the argument as the next argv[i].
				i++;
			} else {

				std::cerr << argv[0]
						<< ": -f option missing config file name."
						<< std::endl;

				exit(1);
			}

		/* 2. job-id */
		} else if (std::string(argv[i]) == "-j") {
			if (i + 1 < argc) {
					jobID = argv[i+1];
					i++;
			}else{
				std::cerr << argv[0]
						 << ": -j option missing job id"
						 << std::endl;
				exit(1);
			}

		/* 3. model parameter file */
		} else if (std::string(argv[i]) == "-m") {
			if (i + 1 < argc) {
				modelFile = argv[i + 1];
				i++;
			} else {
				std::cerr << argv[0]
						<< ": -m option missing model file name"
						<< std::endl;
				exit(1);
			}

		/*4. Layer outputs*/
		} else if(std::string(argv[i]) == "-o"){
			if( i + 1 < argc ){
				std::string out = argv[i+1];
				std::stringstream ss;
				std::string temp;
				ss << out;
				while(!ss.eof()){
					getline(ss,temp,',');
					outputs.push_back(temp);
				}


				i++;

		    }else {

			    std::cerr << argv[0]
				 << ": -o option missing layerName"
				 <<std::endl;

			exit(1);
		    }
		/*5. data file*/
		} else if(std::string(argv[i]) == "-d"){
		    if(i + 1 < argc){
		    	testDataFile = argv[i+1];
		    	i++;
		    }else{
			    std::cerr << argv[0]
				      << ": -o option missing data file"
				      <<std::endl;
			exit(1);

		    }
		/*6. label file*/
		} else if(std::string(argv[i]) == "-l"){
		    if(i + 1 < argc){
		    	testLabelFile = (argv[i+1]);
		    	i++;
		    }else{
			    std::cerr << argv[0]
				      << ": -l option missing label file"
				      <<std::endl;
			    exit(1);
		    }
		/* mean file*/
		} else if (std::string(argv[i]) == "-meanFile"){
			if(i + 1 < argc){
				meanFileName = (argv[i+1]);
				i++;
			}else{
				std::cerr << argv[0]
				          << ": -meanFile option missing mean file"
                          <<std::endl;
				exit(1);
		}
		/* error function */
		}else if(std::string(argv[i]) == "-errorType"){
			if(i + 1 < argc){
				errorType = (argv[i+1]);
				i++;
			}else{
				std::cerr << argv[0]
						  << ": -errorType option missing errorType"
						  <<std::endl;
				exit(1);
			}
		}else {
			std::cerr << argv[0] << ": invalid option -- '"
						<< std::string(argv[i]) << "'" << std::endl;

			printUsage(argv[0]);

			exit(1);
		}
	} // for




	// have we received all the required parameters?
	if (networkCfgFile ==  RUDRA_DEFAULT_STRING){
		std::cerr << argv[0] << " Can not launch a Rudra inference job without network configuration file" << std::endl;
		printUsage(argv[0]);
		exit(1);
	}

	if (modelFile ==  RUDRA_DEFAULT_STRING){
		std::cerr << argv[0] << " Can not launch a Rudra inference job without model parameter file" << std::endl;
		printUsage(argv[0]);
		exit(1);
	}

	if (testDataFile ==  RUDRA_DEFAULT_STRING){
		std::cerr << argv[0] << " Can not launch a Rudra inference job without test data file" << std::endl;
		printUsage(argv[0]);
		exit(1);
	}

	// can we read-in all the files specified by the user?

	std::ofstream fs(networkCfgFile.c_str(),std::ios::in);
	if(!fs){
		std::cerr << argv[0] << " Unable to open network configuration file: " << networkCfgFile << std::endl;
		exit (1);
	}
	fs.close();

	fs.open(modelFile.c_str(), std::ios::in);
	if(!fs){
		std::cerr << argv[0] << " Unable to open model parameter file: " << modelFile << std::endl;
		exit (1);
	}
	fs.close();

	fs.open(testDataFile.c_str(), std::ios::in);
	if(!fs){
		std::cerr << argv[0] << " Unable to open test data file: " << testDataFile << std::endl;
		exit (1);
	}
	fs.close();

	if(testLabelFile != RUDRA_DEFAULT_STRING){
		fs.open(testDataFile.c_str(), std::ios::in);
		if(!fs){
			std::cerr << argv[0] << " Unable to open test label file: " << testLabelFile << std::endl;
			exit (1);
		}
		fs.close();
		doTest = true;

	}


   if(meanFileName != RUDRA_DEFAULT_STRING){
			fs.open(meanFileName.c_str(), std::ios::in);
			if(!fs){
				std::cout << argv[0] << " Unable to open mean file: " << meanFileName << std::endl;
				meanFileName = RUDRA_DEFAULT_STRING;
	}
			fs.close();

	}

   if(errorType != "top-1" && errorType != "top-5"){
	   std::cout << "Unsupported error function: "<< errorType << ". Setting errorType = top-1" << std::endl;
	   errorType = "top-1";
   }


	//rudra::MPILogger::logOnce("Launching NN training with configuration file: " + cfgFile, INFO);
}

void setMLPparams(){
	// 1. Set rudra home
	MLPparams::setRudraHome();
	MLPparams::setMLPdefaults();

	// 2. Set job ID and create working directory
	MLPparams::setJobID(jobID);
	MLPparams::setWD();

	// 3. Set batchSize !!Critical step!!

	int rows, cols;
	MLPparams::readBinHeader(testDataFile,rows,cols);

        MLPparams::_numTestSamples = rows; 
        MLPparams::_numInputDim    = cols;

	// for different models and hardware platforms, we might need different heuristics for selecting the batchSize
	if(rows > 128){
		MLPparams::_batchSize = 128;
	}else{
		MLPparams::_batchSize = rows;
	}

	outputFileName = MLPparams::_logDir + jobID + ".h5";
  	MLPparams::_meanFile = meanFileName;

	MLPparams::_isInference = true;
		
}

void writeOutputFile(Network *nn){

	if(writeOutputs){
	try{
		H5::H5File param_file (outputFileName,H5F_ACC_RDWR);


		for (int i = 0; i < outputs.size(); ++i){
			int layerNum = nn->layerNameMap[outputs[i]];

			H5::DataSet dset(param_file.openDataSet(nn->L[layerNum]->layerName));
			H5::DataSpace dspace = dset.getSpace();
			hsize_t curDims[2];	// current dataset dims
			hsize_t extDims[2]; // extended dims
			dspace.getSimpleExtentDims(curDims,NULL);
                                                         
                        InputPort<float>         port;
                        nn->L[layerNum]->linkXEX(port, CPU);            
                        const Tensor<float>  &outputTensor = port.getTensor(); // nn->L[layerNum]->XEX

			size_t dimM = outputTensor.dimM; // <-- number of rows in a map
			size_t dimN = outputTensor.dimN; // <-- number of cols in a map
			size_t dimK = outputTensor.dimK; // <-- minibatch size
			size_t dimP = outputTensor.dimP; // <-- number of maps

			Matrix<float> tempBuf;
			if(dimK*dimP > 1){
				// the layer has multi-dimensional output (conv/pool etc)

				Matrix<float>res (dimK, dimM * dimN * dimP,_ZEROS); //dimK = MLPparams::_batchSize

					for (uint16 ii = 0; ii < MLPparams::_batchSize; ii++ ){
						// for every sample in the minibatch

						for(uint16 jj = 0; jj < dimP; jj ++){
							// for every map in the sample
							memcpy(res.buf + jj*dimM*dimN + ii*dimM*dimN*dimP, outputTensor.buf[ii*dimP + jj].buf ,sizeof(float) * dimM * dimN);
						}

					}

					tempBuf = res;

			}else{
				tempBuf = outputTensor.buf[0];
			}
			hsize_t dims[2];
			dims[0] = tempBuf.dimM;
			dims[1] = tempBuf.dimN;

			extDims[0] = curDims[0] + dims[0]; 	// extend the number of rows
			extDims[1] = dims[1];				// keep the same number of cols
			dset.extend(extDims);

			//select hyperspace
			dspace = dset.getSpace();
			hsize_t offset[2];
			offset[0] = curDims[0];
			offset[1] = 0;

			dspace.selectHyperslab(H5S_SELECT_SET,dims,offset);
			H5:: DataSpace * memspace = new H5::DataSpace(2, dims, NULL);

			dset.write(tempBuf.buf,H5::PredType::NATIVE_FLOAT, *memspace, dspace);

		}
		param_file.close();

	}
	catch(H5::FileIException &error){
		std::cout << "cannot open file" << std::endl;
		error.printError();
		exit(EXIT_FAILURE);
	}
	catch(H5::DataSetIException &error){
		std::cout << "cannot open dataset" << std::endl;
		error.printError();
		exit(EXIT_FAILURE);
	}
}
}

void initOutputFile(Network * nn){

	// initializes the outputFile
	Logger::logInfo(RUDRA_LINEBREAK);

	// 0.0 sanity check -- do the names given by the user correspond to valid layers?
	std::vector<std::string> temp; // temporary vector to store valid layers
	for (int i =0; i < outputs.size(); ++i){
		if(nn->layerNameMap.find(outputs[i]) == nn->layerNameMap.end()){
			// layerName not found in nn
			Logger::logWarning("RudraInference::Layer " + outputs[i] + " not defined in " + networkCfgFile);

		}else{
			temp.push_back(outputs[i]);
			std::cout << "$$$: " << outputs[i] << std::endl;
		}
	}

	outputs = temp;

	if(outputs.size() == 0){
		rudra::Logger::logWarning("RudraInference::No valid layers specified in -o flag. Layer outputs will not be saved!");
		writeOutputs = false;
	}else{
		writeOutputs = true;
	}

	if(writeOutputs){
		// 1.0: Create h5 file
		const H5std_string FILE_NAME(outputFileName);
		H5::H5File param_file (FILE_NAME,H5F_ACC_TRUNC);

		// 2.0 Create an extensible dataset for each of the layers
		for (int i = 0; i < outputs.size(); ++i){
			int layerNum = nn->layerNameMap[outputs[i]];

                        InputPort<float>         port;
                        nn->L[layerNum]->linkXEX(port, CPU);            
                        const Tensor<float> &outputTensor = port.getTensor(); // nn->L[layerNum]->XEX

			size_t dimM = outputTensor.dimM; // <-- number of rows in a map
			size_t dimN = outputTensor.dimN; // <-- number of cols in a map
			size_t dimK = outputTensor.dimK; // <-- minibatch size
			size_t dimP = outputTensor.dimP; // <-- number of maps

			hsize_t dims[2];
			if(dimK*dimP > 1){
				// the layer has multi-dimensional output (conv/pool etc)

				dims[0] = dimK;
				dims[1] = dimM * dimN * dimP;
			}else{
				dims[0] = dimM; //<--- for fc layers, dimM = MLPparams::_batchSize
				dims[1] = dimN;
			}



			// create chunked dataset
			hsize_t chunk_dims[2];
			chunk_dims[0] = dims[0];
			chunk_dims[1] = dims[1];
			dims[0] = 0;
			H5::DataSpace out (2,dims,maxdims);
			H5::DSetCreatPropList prop;
			prop.setChunk(2, chunk_dims);

			H5::DataSet out_ds(param_file.createDataSet(nn->L[layerNum]->layerName, H5::PredType::NATIVE_FLOAT, out, prop));


		}

		param_file.close();
	}
}

void calcTop1Error(const Matrix<float>& Y, Matrix<float>& labels, int& testErr){
	// computes top-1 error given Y and label
	// Y = output of the softmax layer
	// labels

	size_t dimM = Y.dimM; // batchSize
	size_t dimN = Y.dimN; // sampleDimension
	for(size_t i = 0; i < dimM; ++i) {
			size_t max = 0;
			for (size_t j = 1; j < dimN; ++j) {
				(Y.buf[i*dimN +j] > Y.buf[i*dimN + max])?(max = j):(false);
			}
			//std::cout << "label #" << max << std::endl;
			(labels.buf[i] == max)?(true):(testErr++);
		}
}


void calcTop5Error(const Matrix<float>& Y, Matrix<float>& labels, int& testErr){
	size_t batchSize = Y.dimM; // batchSize
	size_t sampleDim = Y.dimN; // sampleDimension

	for(size_t i = 0; i < batchSize; ++i) {

		// get indices of the top-5 values in Y(i,:)
		size_t indices[5] = {0,1,2,3,4};
		float  val    [5];
		float  minVal;
		int  minIndex=0;

		// [min, index] = min(val);
		val[0] = Y(i,indices[0]);
		minVal = val[0];
		for (int vv = 1; vv < 5; ++vv){
			val[vv] = Y(i,indices[vv]);
			if(val[vv] < minVal){
				minVal   = val[vv];
				minIndex = vv;
			}
		}

		for (size_t j = 5; j < sampleDim; ++j){
			if(Y(i,j) > minVal){
				indices[minIndex] = j;
				//compute minimum value and it's index
				val[0] = Y(i,indices[0]);
				minVal = val[0];
				minIndex = 0;
				for (int vv = 1; vv < 5; ++vv){
					val[vv] = Y(i,indices[vv]);
					if(val[vv] < minVal){
						minVal   = val[vv];
						minIndex = vv;
					}
				}
			}
		} // j : dims in the sample
		bool match = false;
		for (int ll = 0; ll < 5; ++ll){
			if(labels.buf[i] == indices[ll])
				match = true;
		}
		if(!match)
			testErr++;
	} // i : samples


}

int main(int argc, char** argv) {

	Logger::setLoggingLevel(INFO);

	// 1.0 parse the command line
	parseCmdLine(argc, argv);

	// 2.0 set up the relevant values in static class MLPparams
	setMLPparams();

	// 3.0 create NN and load input parameters
	Network* nn = Network::readFromConfigFile(networkCfgFile);
	nn->readParamsH5(modelFile);

	// 4.0 initialize output file
	initOutputFile(nn);
	
	// 5.0 Prepare sequential sample reader

	int numTestSamples, numInputDim;
	MLPparams::readBinHeader(testDataFile, numTestSamples, numInputDim);
	int numMiniBatches = numTestSamples/MLPparams::_batchSize;

	std::cout << testDataFile << std::endl;
	UnifiedBinarySampleSeqReader *binReader = NULL;
	if(doTest){

		binReader = new UnifiedBinarySampleSeqReader(testDataFile, testLabelFile, numTestSamples);

	} else {

		binReader = new UnifiedBinarySampleSeqReader(testDataFile, testDataFile, numTestSamples);
	}

	GPFSSampleClient *sc = new GPFSSampleClient("trainDataReader", true, binReader);

	Matrix<float> minibatchX, minibatchY;

	int testErr = 0;

	// 6.0 set up inference task on all minibatches

	InputPort<float>          port;
	nn->L[nn->N - 2]->linkXEX(port, CPU);
	const Matrix<float>  &outputMatrix = port.getMatrix(); // nn->L[nn->N - 2]->XEX

	std::cout << "Mean File: " << MLPparams::_meanFile << std::endl;
	for (size_t i = 0; i < numMiniBatches; ++i){

		std::cout << "mb# "<< i ;

	    sc->getLabelledSamples(minibatchX, minibatchY);
	    nn->forward(minibatchX);


		if(writeOutputs)
			writeOutputFile(nn);
		if(doTest){
			if(errorType == "top-1"){
				calcTop1Error(outputMatrix, minibatchY, testErr);
			} else if (errorType == "top-5"){
				calcTop5Error(outputMatrix, minibatchY, testErr);
			} else {
				calcTop1Error(outputMatrix, minibatchY, testErr);

			}
		}
		std::cout << " | " << testErr <<  " | " << "% error = "<< float(testErr)/float(numTestSamples)*100 <<std::endl;	
	
	}

	// 7.0 Inference for samples not covered by the minibatches.
	int st = MLPparams::_batchSize * numMiniBatches;
	MLPparams::_batchSize = 1;
	delete nn; delete binReader; delete sc;

	Network* nn1 = Network::readFromConfigFile(networkCfgFile);
	nn1->readParamsH5(modelFile);

	InputPort<float>          port1;
	nn1->L[nn1->N - 2]->linkXEX(port1, CPU);
	const Matrix<float> &outputMatrix1 = port1.getMatrix(); // nn->L[nn->N - 2]->XEX

	UnifiedBinarySampleSeqReader *binReader1 = NULL;
	if(doTest){
		binReader1 = new UnifiedBinarySampleSeqReader(testDataFile, testLabelFile, numTestSamples, st);
	} else {
		binReader1 = new UnifiedBinarySampleSeqReader(testDataFile, testDataFile, numTestSamples, st);
	}

	GPFSSampleClient *sc1 = new GPFSSampleClient("trainDataReader", true, binReader1);


	for (size_t i = st; i < numTestSamples; ++i){

		sc1->getLabelledSamples(minibatchX, minibatchY);
		nn1->forward(minibatchX);

		if(writeOutputs)
			writeOutputFile(nn1);
		if(doTest){
			if(errorType == "top-1"){
				calcTop1Error(outputMatrix1, minibatchY, testErr);
			} else if (errorType == "top-5"){
				calcTop5Error(outputMatrix1, minibatchY, testErr);
			} else {
				calcTop1Error(outputMatrix1, minibatchY, testErr);

			}
		}

	}
	if(doTest){
		Logger::logInfo("TestError = " + convert::T_to_string(float(testErr)/float(numTestSamples) * 100 ) + " %");
	}
	if(writeOutputs){
		Logger::logInfo("Layer outputs stored in " + outputFileName);
	}

	delete nn1; delete binReader1; delete sc1;
	return 0;
}
