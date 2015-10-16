#ifndef __NATIVE_LEARNER_THEANO_H
#define __NATIVE_LEARNER_THEANO_H

#include "learner.h"
#include <cstdlib>
#include <cstddef>
#include <string>
#include <sys/time.h>

namespace rudra {

class GPFSSampleClient;

class NativeLearnerImpl {

public:
	NativeLearnerImpl();
	~NativeLearnerImpl();

	GPFSSampleClient* trainSC;
	void *learner_handle;
	void *learner_data;
	learner_init_t *learner_init;
	learner_destroy_t *learner_destroy;
	learner_netsize_t *learner_netsize;
	learner_train_t *learner_train;
	learner_test_t *learner_test;
	learner_getgrads_t *learner_getgrads;
	learner_accgrads_t *learner_accgrads;
	learner_updatelr_t *learner_updatelr;
	learner_getweights_t *learner_getweights;
	learner_setweights_t *learner_setweights;
	learner_updweights_t *learner_updweights;

	/*
	 * Initialize the network, reading it from the config file.
	 * weightFile: If non-empty, read the weights for the network from the given file.
	 */
	int initNetwork(std::string weightsFile);
};
} // namespace rudra
#endif
