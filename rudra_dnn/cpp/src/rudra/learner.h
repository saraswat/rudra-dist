#ifndef RUDRA_LEARNER_H
#define RUDRA_LEARNER_H

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
int learner_init(void **data, struct param params[], size_t numParams);

/* Deallocate any resource associated with the provided data */
void learner_destroy(void *data);

/* Return the total number of parameters for all the layers (in number
 * of floats).  This is used to size all provided buffers (except
 * train/test data) */
size_t learner_netsize(void *data);

/* Compute cost and accumulate gradients in an internal buffer.
 * Don't update weights.  Return the cost. */
float learner_train(void *data, size_t batchSize,
                   const float *features, ssize_t numInputDims,
                   const float *targets, ssize_t numClasses);

/* Compute cost and return it.  Don't accumulate gradient. Don't
 * update weights */
float learner_test(void *data, size_t batchSize,
                  const float *features, ssize_t numInputDims,
                  const float *targets, ssize_t numClasses);

/* Set the provided buffer to the internal gradients buffer and zero
 * the internal buffer. */
void learner_getgrads(void *data, float *updates);

/* Add the internal gradient buffer to the provided buffer
 * (zero internal?)  */
void learner_accgrads(void *data, float *updates);

/* Set the internal learning rate to the provided value */
void learner_updatelr(void *data, float newLR);

/* Serialize the internal weights into the provided buffer */
void learner_getweights(void *data, float *weights);

/* Deserialized the weights from the provided buffer */
void learner_setweights(void *data, float *weights);

/* Update the internal weights with the provided gradient */
void learner_updweights(void *data, float *grads, size_t numMB);

#ifdef __cplusplus
}
#endif

#endif
