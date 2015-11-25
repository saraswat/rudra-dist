#include "learner.h"
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#include <stdio.h>

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

static int python_refcnt = 0;

static int init_python(void) {
  if (python_refcnt == 0) {
    /* Do not register signal handlers for python.
       Also, this can run more than once safely */
    Py_InitializeEx(0);
    imp_array();
    if (PyErr_Occurred()) {
      PyErr_PrintEx(1);
      return 1;
    }
  }
  python_refcnt++;
  return 0;
}
static void close_python(void) {
  python_refcnt--;
  if (python_refcnt == 0) {
    Py_Finalize();
  }
}

int learner_init(void **_net, struct param params[],
                 size_t numParams) {
  PyObject **net = (PyObject **)_net;
  PyObject *mod = NULL;
  PyObject *pdict = NULL;
  PyObject *init = NULL;
  const char *name = NULL;
  size_t i;

  if (init_python())
    return 1;

  *net = NULL;

  pdict = PyDict_New();
  if (pdict == NULL)
    goto error;

  for (i = 0; i < numParams; i++) {
    PyObject *val = PyString_FromString(params[i].val);
    if (val == NULL)
      goto error;
    if (strcmp(params[i].key, "modelName")==0) {
      name = params[i].val;
    }
    if (PyDict_SetItemString(pdict, params[i].key, val) == -1) {
      Py_DECREF(val);
      goto error;
    }
    Py_DECREF(val);
  }
  fprintf(stdout, "modelName is %s.\n", name);
  if (name == NULL) {
    fprintf(stdout, "modelName name is null: %s\n", name);
    goto error;
  }

  /* This will load the module, but return an error if something else
     is using the import mechanism right now rather than wait
     indefinitely */
  PyErr_Clear(); // Clear error state before returning to Python
  mod = PyImport_ImportModuleNoBlock(name);
  /* A NULL return indicates an error */
  if (mod == NULL) {
    fprintf(stderr, "Could not import module %s, make sure it is available in "
            "the python path.\n", name);
    PyErr_Print();

    goto error;
  }

  init = PyObject_GetAttrString(mod, "myinit");
  if (init == NULL) {
    fprintf(stderr, "No method 'init' found in module %s\n", name);
    PyErr_Print();
    goto error;
  }
  fprintf(stdout, "Method 'myinit' is  %p  \n", init);
    Py_DECREF(mod); mod = NULL;

  PyErr_Clear(); // Clear error state before returning to Python
  *net = PyObject_CallFunctionObjArgs(init, pdict, NULL);
  if (*net == NULL) {
    fprintf(stderr, "Could not create net!\n");
    PyErr_Print();
    goto error;
    exit(1);
  }
  Py_DECREF(init); init = NULL;
  Py_DECREF(pdict); pdict = NULL;

#define CHECK_METHOD(name)                                            \
  if (!PyObject_HasAttrString(*net, name)) {                          \
    fprintf(stderr, "No attribute '%s' found in net object\n", name); \
    goto error;                                                       \
  }
  CHECK_METHOD("size");
  CHECK_METHOD("train");
  CHECK_METHOD("test");
  CHECK_METHOD("get_grads");
  CHECK_METHOD("acc_grads");
  CHECK_METHOD("upd_lr");
  CHECK_METHOD("set_params");
  CHECK_METHOD("get_params");
  CHECK_METHOD("upd_grads");

  return 0;
error:
  PyErr_Clear();
  Py_XDECREF(mod);
  Py_XDECREF(pdict);
  Py_XDECREF(init);
  Py_XDECREF(*net);
  *net = NULL;
  return 1;
}

void learner_destroy(void *net) {
  Py_DECREF((PyObject *)net);
  close_python();
}

size_t learner_netsize(void *net) {
  PyObject *size = PyObject_CallMethod((PyObject *)net, "size", NULL);
  Py_ssize_t res;
  if (size == NULL)
    return 0;
  if (PyErr_Occurred()) {
    PyErr_PrintEx(1);
  }
  if (PyLong_Check(size)) {
    res = PyLong_AsSsize_t(size);
    if (res == -1 && PyErr_Occurred()) {
      PyErr_PrintEx(1);
      res = 0;
    }
#if PY_MAJOR_VERSION < 3
  } else if (PyInt_Check(size)) {
    res = PyInt_AsSsize_t(size);
    if (res == -1 && PyErr_Occurred()) {
      PyErr_PrintEx(1);
      res = 0;
    }
#endif
  } else {
    res = 0;
  }
  Py_DECREF(size);
  return (size_t)res;
}

static float _learner_call2(PyObject *net, const char *meth, size_t batchSize,
                            const float *features, ssize_t numInputDims,
                            const float *targets, ssize_t numClasses) {
  fprintf(stderr, "Making a _learner_call2: %s\n", meth);
  PyObject *fdata = NULL;
  PyObject *tdata = NULL;
  PyObject *res = NULL;
  npy_intp dims[2];
  float error = -1.0f;

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
  dims[0] = batchSize;
  dims[1] = numInputDims;
  fdata = PyArray_New(&PyArray_Type, 2, dims, NPY_FLOAT32, NULL,
                      (void *)features, 4, NPY_ARRAY_IN_ARRAY, NULL);
  if (fdata == NULL || PyErr_Occurred()) {
    fprintf(stderr, "Could not create features array");
    PyErr_PrintEx(1);
    goto error;
  }

  /* Same for targets. */
  dims[1] = numClasses;
  tdata = PyArray_New(&PyArray_Type, 2, dims, NPY_FLOAT32, NULL,
                      (void *)targets, 4, NPY_ARRAY_IN_ARRAY, NULL);
  if (tdata == NULL || PyErr_Occurred()) {
    fprintf(stderr, "Could not create targets array");
    PyErr_PrintEx(1);
    goto error;
  }

  /* Call the train function with the two array objects. */
  PyErr_Clear(); // Clear error state before returning to Python
  res = PyObject_CallMethod((PyObject *)net, (char *)meth, "OO", fdata, tdata);
  if (res == NULL || PyErr_Occurred()) {
    /* We need to care about the return value since python functions
       always return something. The returned object needs to be DECREF'd
       and a NULL return value indicates an error happened. */

    fprintf(stderr, "Error calling %s function", meth);
    PyErr_PrintEx(1);
    goto error;
  }
  /* Make sure the python function did not keep references */
  assert(fdata->ob_refcnt == 1);
  assert(tdata->ob_refcnt == 1);

  error = PyFloat_AsDouble(res);
  if (error == -1.0f && PyErr_Occurred()) {
    PyErr_PrintEx(1);
    /* Yes this does not change anything, but if we ever add code
     * below it reduces the risk of bugs */
    goto error;
  }

  /* Finally release all the objects we created.  This does not
     explicitely destroy theses objects, merely remove the reference
     we had on the since we created them. If there are no other
     references they will be freed. In the case of cycles, collection
     is deferred to when the python GC will run. */
 error:
  PyErr_Clear(); // It is safe to clear error state when there is no error
  Py_XDECREF(fdata);
  Py_XDECREF(tdata);
  Py_XDECREF(res);

  fprintf(stderr, "Finished _learner_call2\n");
  return error;
}

float learner_train(void *net, size_t batchSize,
                    const float *features, ssize_t numInputDims,
                    const float *targets, ssize_t numClasses) {
  return _learner_call2(net, "train", batchSize, features, numInputDims,
                        targets, numClasses);
}

float learner_test(void *net, size_t batchSize,
                   const float *features, ssize_t numInputDims,
                   const float *targets, ssize_t numClasses) {
  return _learner_call2(net, "test", batchSize, features, numInputDims,
                        targets, numClasses);
}

static void _learner_call1(void *net, const char *meth, float *data) {
  PyObject *udata = NULL;
  PyObject *res = NULL;
  npy_intp len = learner_netsize(net);
  fprintf(stderr, "Making _learner_call1: %s\n", meth);
  udata = PyArray_New(&PyArray_Type, 1, &len, NPY_FLOAT32, NULL, data,
                      4, NPY_ARRAY_OUT_ARRAY, NULL);
  fprintf(stdout, "_learner_call1 returned %p.", udata);
  if (udata == NULL || PyErr_Occurred()) {
    // TODO: indicate error?
    fprintf(stdout, "_learner_call1: could not create udata array.\n");
    PyErr_PrintEx(1);
    return;
  }

  PyErr_Clear(); // Clear error state before returning to Python
  res = PyObject_CallMethod((PyObject *)net, (char *)meth, "O", udata);
  /* Make sure the python function did not keep references */
  assert(udata->ob_refcnt == 1);
  Py_DECREF(udata);

  if (res == NULL || PyErr_Occurred()) {
    // TODO: indicate error?
    fprintf(stderr, "Error calling %s function", meth);
    PyErr_PrintEx(1);
    return;
  }
  Py_DECREF(res);
  fprintf(stderr, "Finished _learner_call1: %s\n", meth);
}

void learner_getgrads(void *net, float *updates) {
  _learner_call1(net, "get_grads", updates);
  /*int i;
  float sum = 0;
  for (i = 0; i < 397510; i++) {
    sum += updates[i];
  }
  printf("Sum: %f  ** Avg: %f\n", sum, sum/ 397510.0);*/
}

void learner_accgrads(void *net, float *updates) {
  _learner_call1(net, "acc_grads", updates);
}

void learner_updatelr(void *net, float newLR) {
  PyObject *res;
  res = PyObject_CallMethod((PyObject *)net, "upd_lr", "f", newLR);
  if (res == NULL) {
    PyErr_PrintEx(1);
    // TODO: indicate error
    return;
  }
  Py_DECREF(res);
}

void learner_getweights(void *net, float *weights) {
  _learner_call1(net, "get_params", weights);
}

void learner_setweights(void *net, float *weights) {
  _learner_call1(net, "set_params", weights);
}

void learner_updweights(void *net, float *grads, size_t numMB) {
  PyObject *gdata = NULL;
  PyObject *res = NULL;
  npy_intp len = learner_netsize(net);

  gdata = PyArray_New(&PyArray_Type, 1, &len, NPY_FLOAT32, NULL, grads,
                      4, NPY_ARRAY_OUT_ARRAY, NULL);
  if (gdata == NULL) {
    // TODO: indicate error?
    PyErr_PrintEx(1);
    return;
  }

  PyErr_Clear(); // Clear error state before returning to Python
  res = PyObject_CallMethod((PyObject *)net, "upd_grads", "On", gdata,
                            (Py_ssize_t)numMB);
  /* Make sure the python function did not keep references */
  assert(gdata->ob_refcnt == 1);
  Py_DECREF(gdata);

  if (res == NULL) {
    // TODO: indicate error?
    PyErr_PrintEx(1);
    return;
  }
  Py_DECREF(res);
}
