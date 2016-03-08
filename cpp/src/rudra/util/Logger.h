/*
 * Logger.h
 *
 * Licensed Materials - Property of IBM
 *
 * Rudra Distributed Learning Platform
 *
 *  Copyright IBM Corp. 2016 All Rights Reserved
 */

#ifndef RUDRA_UTIL_LOGGER_H_
#define RUDRA_UTIL_LOGGER_H_
#include <fstream>

enum LogLevel {
	INFO, WARNING, ERROR, FATAL
};
static std::string levelName[] = { "INFO", "WARNING", "ERROR", "FATAL" };

static  LogLevel LOG_LEVEL = WARNING;

namespace rudra {
class Logger {
public:
	static void setLogFile(std::string fname);
	static void setLoggingLevel(int i);
	static void log(std::string msg, LogLevel level); // log message at chosen level of severity
	static void logInfo(std::string msg);
	static void logWarning(std::string msg);
	static void logError(std::string msg);
	static void logFatal(std::string msg); // log error and terminate the program
	static void dumpTable(std::string fileName, float **table, int m, int n);
};

}
#endif /* RUDRA_UTIL_LOGGER_H_ */
