#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#include <stdio.h>
#include "theano.h"

static PyObject *mod = NULL;

/* This is required because import_array is a macro that may return */
static
#if PY_MAJOR_VERSION == 3
PyObject *
#else
void
#endif
imp_array(void) {
  import_array();
}

int theano_init(void) {
  const char *name = "mlp";

  /* Do not register signal handlers for python.
     Also, this can run more than once safely */
  Py_InitializeEx(0);
  imp_array();
  if (PyErr_Occurred()) {
    PyErr_Clear();
    return -1;
  }

  /* Make sure to clear any previous loaded module */
  Py_XDECREF(mod);
  /* This will load the module, but return an error if something else
     is using the import mechanism right now rather than wait
     indefinitely */
  mod = PyImport_ImportModuleNoBlock(name);
  /* A NULL return indicates an error */
  if (mod == NULL) {
    fprintf(stderr, "Could not import module %s, make sure it is available in "
            "the python path.\n", name);
    /* We need to clear the python error otherwise it may show up
       again in some other places. */
    PyErr_Clear();
    return -1;
  }

  /* This is just preemptive error checking to make sure the module
     has the methods that we need. */
  if (!PyObject_HasAttrString(mod, "train")) {
    fprintf(stderr, "No method 'train' found in module\n");
    Py_XDECREF(mod);
    mod = NULL;
    return -1;
  }
  if (!PyObject_HasAttrString(mod, "set_params")) {
    fprintf(stderr, "No method 'set_params' found in module\n");
    Py_XDECREF(mod);
    mod = NULL;
    return -1;
  }
  if (!PyObject_HasAttrString(mod, "get_updates")) {
    fprintf(stderr, "No method 'get_updates' found in module\n");
    Py_XDECREF(mod);
    mod = NULL;
    return -1;
  }
  return 0;
}

void theano_fini(void) {
  /* Make sure to DECREF our module (XDECREF will check if it's a NULL
   * pointer and do nothing in that case) to properly release memory.
   */
  Py_XDECREF(mod);
  mod = NULL;
  Py_Finalize();
}


float theano_train(const float *features, int fnd, ssize_t *fdims,
           const float *targets, int tnd, ssize_t *tdims) {
  PyGILState_STATE gstate;
  PyObject *train = NULL;
  PyObject *fdata = NULL;
  PyObject *tdata = NULL;
  PyObject *res = NULL;

  /* If no module was loaded there is nothing to do */
  if (mod == NULL) return 1.0;

  /* This makes sure we have the GIL and can run python methods (that
     includes the C-API part of python). This version is safe even if
     your program has multiple threads. */
  gstate = PyGILState_Ensure();

  /* Get a reference to the "train" method */
  train = PyObject_GetAttrString(mod, "train");
  if (train == NULL) {
    fprintf(stderr, "Could not get train function from module");
    goto error;
  }

  /* Create an ndarray object that points to the features. We don't
     specify flags that make the array writeable to prevent the python
     code from modifying our data. This could also be enforced by just
     making sure the python code does not write to the data.

     This makes a reference to our data, not a copy. In the case of
     this script, it is safe since we know that the python code does
     not keep any reference to these arrays. If the python code needs
     to keep a reference we would need to make a copy to ensure the
     validity of the data pointers for the lifetime of the objects.

     We pass NULL for the strides which makes it assume
     C-contiguous. If the data was arranged differently we would have
     to pass explicit strides. */
  fdata = PyArray_New(&PyArray_Type, fnd, fdims, NPY_FLOAT32, NULL,
                      (void *)features, 4, NPY_ARRAY_IN_ARRAY, NULL);
  if (fdata == NULL) {
    fprintf(stderr, "Could not create features array");
    goto error;
  }


  /* Same for targets. */
  tdata = PyArray_New(&PyArray_Type, tnd, tdims, NPY_FLOAT32, NULL,
                      (void *)targets, 4, NPY_ARRAY_IN_ARRAY, NULL);
  if (tdata == NULL) {
    fprintf(stderr, "Could not create targets array");
    goto error;
  }

  /* Call the train function with the two array objects. */
  res = PyObject_CallFunctionObjArgs(train, fdata, tdata, NULL);
  if (res == NULL)
    fprintf(stderr, "Error calling train function");
  /* We need to care about the return value since python functions
     always return something. The returned object needs to be DECREF'd
     and a NULL return value indicates an error happened. */

  /* Finally release all the objects we created.  This does not
     explicitely destroy theses objects, merely remove the reference
     we had on the since we created them. If there are no other
     references they will be freed. In the case of cycles, collection
     is deferred to when the python GC will run. */
 error:
  PyErr_Clear(); // It is safe to clear error state when there is no error
  Py_XDECREF(train);
  Py_XDECREF(fdata);
  Py_XDECREF(tdata);
  Py_XDECREF(res);

  /* Finally release the GIL to let other code run */
  PyGILState_Release(gstate);
  // vj to do: get the training error from theano and return it.
  return 1.0f;
}

float theano_train_entry(ssize_t batchSize, const float *features, ssize_t numInputDims, 
			const float *targets, ssize_t numClasses) {
  ssize_t fdims[2] = {batchSize, numInputDims}; // hmm do these need to be in reverse order?
  ssize_t tdims[2] = {batchSize, numClasses};
  return theano_train(features,2, fdims, targets, 2, tdims);


}
/* These must macth to what is defined inside of the python file */
static npy_intp W1_shape[2] = {784, 500};
static npy_intp b1_shape[1] = {500};
static npy_intp W2_shape[2] = {500, 10};
static npy_intp b2_shape[1] = {10};

/* This includes the element size */
static const size_t param_sizes[] = {
  784 * 500 * 4, // W1
  500 * 4, // b1
  500 * 10 * 4, // W2
  10 * 4 // b2
};

size_t theano_networkSize(void) {
  return param_sizes[0] + param_sizes[1] + param_sizes[2] + param_sizes[3];
}

void theano_set_param(const void *p, size_t size) {
  PyGILState_STATE gstate;
  PyObject *set_data = NULL;
  PyObject *W1 = NULL;
  PyObject *b1 = NULL;
  PyObject *W2 = NULL;
  PyObject *b2 = NULL;
  PyObject *res = NULL;
  char *ptr = (char *)p;

  if (mod == NULL) return;

  gstate = PyGILState_Ensure();

  set_data = PyObject_GetAttrString(mod, "set_params");
  if (set_data == NULL) {
    fprintf(stderr, "Could not get set_params function from module");
    goto error;
  }

  /* We split the data into multiple arrays in C since it's much
     easier. */
  W1 = PyArray_New(&PyArray_Type, 2, W1_shape, NPY_FLOAT32, NULL, ptr,
                   4, NPY_ARRAY_IN_ARRAY, NULL);
  if (W1 == NULL) {
    fprintf(stderr, "Could not create W1 array");
    goto error;
  }
  ptr += param_sizes[0];

  b1 = PyArray_New(&PyArray_Type, 1, b1_shape, NPY_FLOAT32, NULL, ptr,
                   4, NPY_ARRAY_IN_ARRAY, NULL);
  if (b1 == NULL) {
    fprintf(stderr, "Could not create b1 array");
    goto error;
  }
  ptr += param_sizes[1];

  W2 = PyArray_New(&PyArray_Type, 2, W2_shape, NPY_FLOAT32, NULL, ptr,
                   4, NPY_ARRAY_IN_ARRAY, NULL);
  if (W2 == NULL) {
    fprintf(stderr, "Could not create W2 array");
    goto error;
  }
  ptr += param_sizes[2];

  b2 = PyArray_New(&PyArray_Type, 1, b2_shape, NPY_FLOAT32, NULL, ptr,
                   4, NPY_ARRAY_IN_ARRAY, NULL);
  if (b2 == NULL) {
    fprintf(stderr, "Could not create b2 array");
    goto error;
  }

  res = PyObject_CallFunctionObjArgs(set_data, W1, b1, W2, b2, NULL);
  if (res == NULL)
    fprintf(stderr, "Error calling set_params function");

 error:
  PyErr_Clear();
  Py_XDECREF(set_data);
  Py_XDECREF(W1);
  Py_XDECREF(b1);
  Py_XDECREF(W2);
  Py_XDECREF(b2);
  Py_XDECREF(res);

  PyGILState_Release(gstate);
}
void theano_set_param_entry(const void *p) {
  theano_set_param(p, theano_networkSize());
}

void theano_get_update(void *p, size_t size) {
  PyGILState_STATE gstate;
  PyObject *fetch_update = NULL;
  PyObject *W1 = NULL;
  PyObject *b1 = NULL;
  PyObject *W2 = NULL;
  PyObject *b2 = NULL;
  PyObject *res = NULL;
  char *ptr = p;

  if (mod == NULL) return;

  gstate = PyGILState_Ensure();

  fetch_update = PyObject_GetAttrString(mod, "get_updates");
  if (fetch_update == NULL) {
    fprintf(stderr, "Could not get get_updates function from module");
    goto error;
  }

  /* Here we do specify flags that make the array writeable since we
     want the python code to write its data there. */
  W1 = PyArray_New(&PyArray_Type, 2, W1_shape, NPY_FLOAT32, NULL, ptr,
                   4, NPY_ARRAY_OUT_ARRAY, NULL);
  if (W1 == NULL) {
    fprintf(stderr, "Could not create W1 array");
    goto error;
  }
  ptr += param_sizes[0];

  b1 = PyArray_New(&PyArray_Type, 1, b1_shape, NPY_FLOAT32, NULL, ptr,
                   4, NPY_ARRAY_OUT_ARRAY, NULL);
  if (b1 == NULL) {
    fprintf(stderr, "Could not create b1 array");
    goto error;
  }
  ptr += param_sizes[1];

  W2 = PyArray_New(&PyArray_Type, 2, W2_shape, NPY_FLOAT32, NULL, ptr,
                   4, NPY_ARRAY_OUT_ARRAY, NULL);
  if (W2 == NULL) {
    fprintf(stderr, "Could not create W2 array");
    goto error;
  }
  ptr += param_sizes[2];

  b2 = PyArray_New(&PyArray_Type, 1, b2_shape, NPY_FLOAT32, NULL, ptr,
                   4, NPY_ARRAY_OUT_ARRAY, NULL);
  if (b2 == NULL) {
    fprintf(stderr, "Could not create b2 array");
    goto error;
  }

  res = PyObject_CallFunctionObjArgs(fetch_update, W1, b1, W2, b2, NULL);
  if (res == NULL)
    fprintf(stderr, "Error calling get_updates function");  

 error:
  PyErr_Clear();
  Py_XDECREF(fetch_update);
  Py_XDECREF(W1);
  Py_XDECREF(b1);
  Py_XDECREF(W2);
  Py_XDECREF(b2);
  Py_XDECREF(res);

  PyGILState_Release(gstate);
}

void theano_get_update_entry(void *p) {
  theano_get_update(p, theano_networkSize());
}

/*int main(void) {
  fprintf(stdout, "Hello, world...");
  init();
  fini();
  fprintf(stdout, "\n...done!");
  return 1;
}
*/
