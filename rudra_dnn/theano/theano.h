#ifndef __THEANO_H_
#define __THEANO_H_

int theano_init(void);
void theano_fini(void);
size_t theano_networkSize(void);
void ensureLoaded(float*);
float theano_train_entry(ssize_t batchSize, const float *features, ssize_t numInputDims, 
			const float *targets, ssize_t numClasses);
void theano_set_param_entry(const void *p);
void theano_get_update_entry(void *p);

#endif /* __THEANO_H_*/
