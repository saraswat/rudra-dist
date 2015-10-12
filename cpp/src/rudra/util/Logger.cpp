#include "rudra/util/Logger.h"
#include <cstdlib>
#include <iostream>

namespace rudra {
void Logger::setLogFile(std::string f) {
	// TODO allow logging to file
	/*
	 std::string fName = f;
	 std::cout << "Opening log file " << f << std::endl;
	 _output.open(fName.c_str(), std::ios::trunc);
	 if (_output.bad()) {
	 std::cerr << "Error opening log file: " << fName << std::endl;
	 exit(1);
	 }
	 */
}

    void Logger::setLoggingLevel(int i){
	switch(i){
	case 1:
	    LOG_LEVEL = INFO;
	    break;
	case 2:
	    LOG_LEVEL = WARNING;
	    break;
	case 3:
	    LOG_LEVEL = ERROR;
	    break;
	case 4:
	    LOG_LEVEL = FATAL;
	    break;
	default:
	    LOG_LEVEL = INFO;
	    break;
	}
    
    }

void Logger::log(std::string msg, LogLevel level) {
	if (LOG_LEVEL <= level) {
		// TODO: prefix with level?
		std::cout << msg << std::endl;
	}
}

void Logger::logInfo(std::string msg) {
	log(msg, INFO);
}

void Logger::logWarning(std::string msg) {
	log(msg, WARNING);
}

void Logger::logError(std::string msg) {
	// TODO copy to log file
	//log(msg, ERROR);
	if (LOG_LEVEL <= ERROR) {
		std::cerr << msg << std::endl;
	}
}

void Logger::logFatal(std::string msg) {
	// TODO copy to log file
	//log(msg, FATAL);
	std::cerr << msg << std::endl;
	exit(EXIT_FAILURE);
}
    /**
     *@param fileName the file we dump stats to
     *@param table stats
     *@param m, n dimensions of stats
     */
    void Logger::dumpTable(std::string fileName, float **table, int m, int n){
	std::ofstream f1(fileName.c_str(), std::ios::trunc); //create that file
	for(int i = 0; i < m; ++i){
	    for(int j = 0; j < n; ++j){
		f1 << table[i][j]<< "\t";
	    }
	    f1<<std::endl; // end of line and flushes a stream
	}
	f1.close();
    }
} /* namespace rudra */
