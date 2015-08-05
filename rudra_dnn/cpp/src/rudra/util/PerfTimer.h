/*
 * PerfTimer.h
 *
 * A simple performance timer.
 */

#include <string>
#include <sys/time.h>

#ifndef RUDRA_UTIL_PERF_TIMER_H_
#define RUDRA_UTIL_PERF_TIMER_H_

// learner side
#define EID_LOAD_DATA 1
#define EID_BCAST_WEIGHTS 2
#define EID_DESER_WEIGHTS 3
#define EID_SEL_TRAIN_DATA 4
#define EID_PULL_WEIGHTS 5
#define EID_TRAIN 6
#define EID_SER_UPDATES 7
#define EID_PUSH_UPDATES 8
#define EID_REPORT_TRAIN_ERR 9
#define EID_TEST 10
#define EID_REPORT_TEST_DATA 11

// param server side
#define EID_DESER_UPDATES 12
#define EID_SUM_UPDATES 13
#define EID_APPLY_UPDATES 14
#define EID_SER_WEIGHTS 15
#define EID_SEND_WEIGHTS 16

namespace rudra {
static std::string LOG_SEP = ":";

class PerfTimer {
public:
  PerfTimer(int pid, int eid, std::string opName = "", std::string layerName = "");
	PerfTimer();
        PerfTimer(const PerfTimer &t);
	~PerfTimer();
        PerfTimer &operator=(const PerfTimer &t);
        void setName(std::string opName, std::string layerName);
	void tic();
	void toc();
        void addDelay(float delay);
	void logLocalMsg(int epochNum, int mbNum);
	float getDelta() const;
	float getDeltams() const;
private:
	int pid;
	int eid;
	struct timeval start;
	struct timeval end;
                 

        std::string   opName;
        std::string   layerName;
        float         minMs;
        float         maxMs;
        float         totalMs;
        unsigned long totalTics;
                 
        static std::string lastLayerNamePrinted;
};
} /* namespace rudra */
#endif /* RUDRA_UTIL_PERF_TIMER_H_ */
