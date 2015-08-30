/*
 * PerfTimer.cpp
 */

#include "rudra/util/PerfTimer.h"
#include "rudra/util/Logger.h"
#include "rudra/util/Checking.h"
#include <float.h>
#include <stdlib.h>
#include <sstream>
#include <iostream>     // std::cout, std::right, std::endl
#include <iomanip> 

namespace rudra {


  ////////////////////////////////////////////////////////////////////////
  //
  //    Stats
  //
  ////////////////////////////////////////////////////////////////////////


  PerfTimer::Stats::Stats()
  {       
    minMs     = FLT_MAX;
    maxMs     = 0;
    totalMs   = 0;
    totalTics = 0; 
  }


  PerfTimer::Stats::Stats(const PerfTimer::Stats &t)
  {
    minMs     = t.minMs;
    maxMs     = t.maxMs;
    totalMs   = t.totalMs;
    totalTics = t.totalTics;
  }

  PerfTimer::Stats &PerfTimer::Stats::operator=(const PerfTimer::Stats &t) {
    minMs     = t.minMs;
    maxMs     = t.maxMs;
    totalMs   = t.totalMs;
    totalTics = t.totalTics;
    return *this;
  }


  void PerfTimer::Stats::addDelay(float delay) {
    totalMs  += delay;
    totalTics++;
    if (minMs > delay) minMs = delay;
    if (maxMs < delay) maxMs = delay;
  }


  void PerfTimer::Stats::display(std::stringstream &msg,
                                 std::string        layerName,
                                 std::string        opName,
                                 bool               startLayer)
  {
    // Is this the first call to print timing for layerName?
    if (startLayer) {
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
  }


  ////////////////////////////////////////////////////////////////////////
  //
  //    CriticalPath
  //
  ////////////////////////////////////////////////////////////////////////
                 
  // This is a data structure collecting stats about operations declared
  // critical.
  // It is based on the assumption that all operations declared critical
  // are executed the same number of times; so the n-th invocations of
  // two critical operations must belong to the same execution of the critical path.
  class CriticalPath
  {
  private:
    PerfTimer::Stats stats;        // collected for executions of critical path
    
    // We use a queue to deal with this problem:
    // Delay for an operation on a later   execution of the critical path may come before
    // delay for an operation on a earlier execution of the critical path.
    // This is caused by asynchronous behavior of the GPU.
    // Therefore Q_SIZE must be as least as large as QUEUE_SIZE in GPUtimer.
    static const int Q_SIZE = 10;
    float           *delayQ;        // array of accumulated delays for several executions of the critical path
    int              numInQ;        // number of critical paths placed into delayQ
    int              numProcessed;  // number of critical paths processed and removed from delayQ

  public:
    CriticalPath() :
      stats()
    {
      delayQ = (float *) malloc(Q_SIZE * sizeof(float));
      numInQ = numProcessed = 0;
    }
    
    ~CriticalPath()
    {
      // Process any left-over critical paths in delayQ
      for (; numProcessed < numInQ; ++numProcessed) {
        stats.addDelay(delayQ[numProcessed % Q_SIZE]);
      }
      
      // Final report
      if (numProcessed > 0) {
        std::stringstream msg;
        stats.display(msg, "Network", "critical path", false);
        Logger::logInfo(msg.str());
      }
      
      free(delayQ);
    }
    
    // The given critical path finished execution.
    // Add its delay to the stats
    void addDelay(float delay,
                  int   whichCriticalPath)
    {
      // If there is not enough room in delayQ for whichCriticalPath,
      // make room by processing earlier critical paths
      while(whichCriticalPath - numProcessed >= Q_SIZE) {
        stats.addDelay(delayQ[numProcessed % Q_SIZE]);
        ++numProcessed;
        RUDRA_CHECK(numProcessed < numInQ, RUDRA_VAR(numProcessed) << RUDRA_VAR(numInQ));
      }
      
      // If this is the first operation of a new critical path, update numInQ
      for (; numInQ <= whichCriticalPath; ++numInQ) {
        delayQ[numInQ % Q_SIZE] = 0;                  // initialize for new critical path
      }

      delayQ[whichCriticalPath % Q_SIZE] += delay;  // add contribution of this delay
    }
  };


  // At the end of the run we rely on the destructor to be called
  // to display the statistics of critical path executions
  static CriticalPath criticalPath;



  ////////////////////////////////////////////////////////////////////////
  //
  //    PerfTimer
  //
  ////////////////////////////////////////////////////////////////////////

  // This static variable is used to make printout more readable.
  // Timing is printed from destructor, which gets invoked when a layer
  // is destructed, so all the timers in one layer get destructed together.
  // This variable lets us know when timers for a new layer start to be destructed.
  std::string PerfTimer::lastLayerNamePrinted;


  PerfTimer::PerfTimer(int         pid, 
                       int         eid, 
                       std::string opName,
                       std::string layerName) :
    stats(),
    pid      (pid), 
    eid      (eid) 
  {
    setName(opName, layerName);
    isCritical      = false;  // possibly set by caling setCritical()
  }
                 
  PerfTimer::PerfTimer() :
    stats(),
    pid(0),
    eid(0),
    opName(""),
    layerName("")
  {
    isCritical      = false;  // possibly set by caling setCritical()
  }

  PerfTimer::PerfTimer(const PerfTimer &t) :
    stats     (t.stats),
    pid       (t.pid),
    eid       (t.eid),
    opName    (t.opName),
    layerName (t.layerName)
  {
    isCritical      = t.isCritical;
  }

  PerfTimer &PerfTimer::operator=(const PerfTimer &t) {
    pid       = t.pid;
    eid       = t.eid;
    opName    = t.opName;
    layerName = t.layerName;
    stats     = t.stats;
    isCritical      = t.isCritical;
    return *this;
  }
                 
  PerfTimer::~PerfTimer()
  {
    if (stats.totalTics > 0 && opName != "" && layerName != "") {
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
    
      stats.display(msg, layerName, opName, layerName != lastLayerNamePrinted);
                                                         
      if (isCritical)      msg << "  <-- critical";

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

  void PerfTimer::setCritical()
  {
    this->isCritical      = true;
  }

  void PerfTimer::tic() {
    gettimeofday(&start, NULL);
  }

  void PerfTimer::toc() {
    gettimeofday(&end, NULL);
    addDelay(getDeltams());
  }
                 
                 
  void PerfTimer::addDelay(float delay) {
    if (isCritical) {
      criticalPath.addDelay(delay, stats.totalTics);
    }
    stats.addDelay(delay);
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
