/*
 * PerfTimer.cpp
 */

#include "rudra/util/PerfTimer.h"
#include "rudra/util/Logger.h"
#include <float.h>
#include <sstream>
#include <iostream>     // std::cout, std::right, std::endl
#include <iomanip> 

namespace rudra {

  // This static variable is used to make printout more readable.
  // Timing is printed from destructor, which gets invoked when a layer
  // is destructed, so all the timers in one layer get destructed together.
  // This variable lets us know when timers for a new layer start to be destructed.
  std::string PerfTimer::lastLayerNamePrinted;


  PerfTimer::PerfTimer(int         pid, 
                       int         eid, 
                       std::string opName,
                       std::string layerName) :
  pid      (pid), 
  eid      (eid) {
    setName(opName, layerName);
    minMs = FLT_MAX;
    maxMs = 0;
    totalMs = 0;
    totalTics = 0;   
  }
                 
  PerfTimer::PerfTimer() :
  pid(0),eid(0),opName(""),layerName(""){
    minMs = FLT_MAX;
    maxMs = 0;
    totalMs = 0;
    totalTics = 0;   
  }

  PerfTimer::PerfTimer(const PerfTimer &t) :
  pid       (t.pid),
  eid       (t.eid),
  opName    (t.opName),
  layerName (t.layerName){
    minMs = t.minMs;
    maxMs = t.maxMs;
    totalMs = t.totalMs;
    totalTics = t.totalTics;
  }

  PerfTimer &PerfTimer::operator=(const PerfTimer &t) {
    pid       = t.pid;
    eid       = t.eid;
    opName    = t.opName;
    layerName = t.layerName;
    minMs     = t.minMs;
    maxMs     = t.maxMs;
    totalMs   = t.totalMs;
    totalTics = t.totalTics;
    return *this;
  }
                 
PerfTimer::~PerfTimer()
{
  if (totalTics > 0 && opName != "" && layerName != "") {
    std::stringstream msg;

    // Is this the first call to print timing?
    if (lastLayerNamePrinted == "") {
      msg << std::right << std::setw(50) << "Final Timing Of Each Operation" << std::endl
          << std::left  << std::setw(25) << "OPERATION "
          << std::left  << std::setw( 5) << "LAYER"            << "    " 
          << std::right << std::setw( 8) << "CALLS"            << "    "
          << std::right << std::setw(15) << "TOTAL TIME" 
          << std::right << std::setw(15) << "MIN TIME "   
          << std::right << std::setw(15) << "AVG TIME "   
          << std::right << std::setw(15) << "MAX TIME "
          << std::endl  << std::endl;
    }
    
    // Is this the first call to print timing for layerName?
    if (layerName != lastLayerNamePrinted) {
      msg << std::left  << std::setw(25) << "---------"
          << std::left  << std::setw( 5) << layerName
          << std::endl;
    }
               
    msg << std::left  << std::setw(25) << opName 
        << std::left  << std::setw( 5)                                       << " "               << "    " 
        << std::right << std::setw( 8)                         << std::fixed << totalTics         << "    " 
        << std::right << std::setw(11) << std::setprecision(2) << std::fixed << totalMs/1000      << " sec"
        << std::right << std::setw(11) << std::setprecision(2) << std::fixed << minMs             << " ms "
        << std::right << std::setw(11) << std::setprecision(2) << std::fixed << totalMs/totalTics << " ms "
        << std::right << std::setw(11) << std::setprecision(2) << std::fixed << maxMs             << " ms "
      ;

    Logger::logInfo(msg.str());
                                                         
    lastLayerNamePrinted = layerName;  
  }
}



  void PerfTimer::setName(std::string opName, 
                          std::string layerName)
  {
    this->opName    = opName;
    this->layerName = layerName;
  }

void PerfTimer::tic() {
	gettimeofday(&start, NULL);
}

void PerfTimer::toc() {
	gettimeofday(&end, NULL);
        addDelay(getDeltams());
}
                 
void PerfTimer::addDelay(float delay) {
        totalMs  += delay;
        totalTics++;
        if (minMs > delay) minMs = delay;
        if (maxMs < delay) maxMs = delay;
}
                 

float PerfTimer::getDelta() const {
	time_t startSec = start.tv_sec;
	suseconds_t startUSec = start.tv_usec;
	time_t endSec = end.tv_sec;
	suseconds_t endUSec = end.tv_usec;
	return float(((endSec - startSec) * 1e6 + (endUSec - startUSec)) / 1e6);
}

float PerfTimer::getDeltams() const{
	time_t startSec = start.tv_sec;
	suseconds_t startUSec = start.tv_usec;
	time_t endSec = end.tv_sec;
	suseconds_t endUSec = end.tv_usec;
	return float(((endSec - startSec) * 1e6 + (endUSec - startUSec)) / 1e3);
}

void PerfTimer::logLocalMsg(int epochNum, int mbNum) {
	std::stringstream logMsg;
	logMsg << "P" << pid << LOG_SEP << "E" << eid << LOG_SEP << getDelta()
			<< LOG_SEP << epochNum << LOG_SEP << mbNum;
	Logger::logInfo(logMsg.str());
}
} /* namespace rudra */
