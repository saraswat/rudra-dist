#ifndef RUDRA_LEARNER_H
#define RUDRA_LEARNER_H

#include <sys/types.h>

#ifdef __cplusplus
extern "C" {
#endif
#ifdef CONFUSE_EMACS
}
#endif

struct param {
  const char *key;
  const char *val;
};


/* Initialize the learner.  Return 0 for success and 1 for failure */
typedef int learner_init_t(void **data, struct param params[],
                           size_t numParams);
learner_init_t learner_init;

/* Deallocate any resource associated with the provided data */
typedef void learner_destroy_t(void *data);
learner_destroy_t learner_destroy;

/* Return the total number of parameters for all the layers (in number
 * of floats).  This is used to size all provided buffers (except
 * train/test data) */
typedef size_t learner_netsize_t(void *data);
learner_netsize_t learner_netsize;

/* Compute cost and accumulate gradients in an internal buffer.
 * Don't update weights.  Return the cost. */
typedef float learner_train_t(void *data, size_t batchSize,
                              const float *features, ssize_t numInputDims,
                              const float *targets, ssize_t numClasses);
learner_train_t learner_train;

/* Compute cost and return it.  Don't accumulate gradient. Don't
 * update weights */
typedef float learner_test_t(void *data, size_t batchSize,
                             const float *features, ssize_t numInputDims,
                             const float *targets, ssize_t numClasses);
learner_test_t learner_test;

/* Set the provided buffer to the internal gradients buffer and zero
 * the internal buffer. */
typedef void learner_getgrads_t(void *data, float *updates);
learner_getgrads_t learner_getgrads;

/* Add the internal gradient buffer to the provided buffer
 * (zero internal?)  */
typedef void learner_accgrads_t(void *data, float *updates);
learner_accgrads_t learner_accgrads;

/* Multiply the initial learning rate by the provided value */
typedef void learner_setlrmult_t(void *data, float lrMult);
learner_setlrmult_t learner_setlrmult;

/* Serialize the internal weights into the provided buffer */
typedef void learner_getweights_t(void *data, float *weights);
learner_getweights_t learner_getweights;

/* Deserialized the weights from the provided buffer */
typedef void learner_setweights_t(void *data, float *weights);
learner_setweights_t learner_setweights;

/* Update the internal weights with the provided gradient */
typedef void learner_updweights_t(void *data, float *grads, const float multiplier);
learner_updweights_t learner_updweights;

#ifdef __cplusplus
}
#endif

#endif
