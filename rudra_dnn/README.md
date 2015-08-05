# Rudra Distributed Learning Platform                         {#mainpage}

Rudra is a distributed framework for large-scale machine learning, which accepts training data and model configuration as inputs from the user and outputs the parameters of the trained model.

Detailed documentation on input formats, invocation, sample datasets and planned features can be found on the [project wiki](https://github.rtp.raleigh.ibm.com/rudra/rudra_dnn/wikis/home).

# Installation

Dependencies:
1. g++ (4.4.7 or higher) / xlC  
2. BLAS library: [OpenBLAS](http://openblas.org), [ATLAS](http://math-atlas.sourceforge.net/), ESSL (IBM Power) or NetLIB
3. [HDF5 library](https://www.hdfgroup.org/HDF5/) (system installation or installed in user directory by the `install_hdf5.sh` shell script)

To build:

    $ cd cpp
    $ ./install_hdf5.sh
    $ source ./../rudra.profile
    $ make rudra_standalone rudra_inference
    $ make rudra

Note:
1. `./install_hdf5.sh` needs to be executed only during the first time setup. 

2. `rudra.profile` sets the necessary environment variables needed for building and running Rudra. Amongst other things, it sets the `$RUDRA_HOME` environment variable. In some cases, you may need to modify `rudra.profile` to correctly point to your local BLAS installations. 

3. After successful build, three binaries will appear in `$RUDRA_HOME/cpp`: 

 1. `rudra_standalone`: Launches training job on a single node, no MPI calls.
 2. `rudra_inference` : Used for feed-forward/inference on the test data. Runs on a single node, no MPI calls.
 3. `rudra` : Rudra distributed, used to launch MPI jobs on the cluster

4. For Rudra installation on isolated machines (such as your personal linux machine/ laptop), you may choose to skip the last step: `make rudra`.

# Documentation

Dependencies: [Doxygen](http://www.stack.nl/~dimitri/doxygen/) v1.8.0 or higher

    $ cd doc && doxygen

