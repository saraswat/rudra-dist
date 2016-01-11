# Rudra Distributed Learning Platform                         {#mainpage}

Rudra is a distributed framework for large-scale machine learning, which accepts training data and model configuration as inputs from the user and outputs the parameters of the trained model.

Detailed documentation on input formats, invocation, sample datasets and planned features can be found on the [project wiki](https://github.com/saraswat/rudra-dist/wiki).

# Installation

Dependencies:

1. [X10](http://x10-lang.org/) (version 2.5.4 or higher)
2. g++ (4.4.7 or higher) / xlC
3. (optional - for Theano learner) [Theano](http://deeplearning.net/software/theano/) plus [Theano prerequisites](http://deeplearning.net/software/theano/install.html#requirements)

The default version of Rudra uses a 
[Theano](http://deeplearning.net/software/theano/) learner, the source code
for which is included in this package.
A mock learner is also included for unit testing purposes.
Other learners are supported by implementing the learner API in 
`include/NativeLearner.h` . The make variable `RUDRA_LEARNER` chooses between
different learner implementations e.g. basic, theano, mock.
Setting `RUDRA_LEARNER=xxx` requires the build to link against a learner
implementation at `lib/librudralearner-xxx.so`.

To build the default (Theano) version of Rudra, simply run:

    $ source rudra.profile
    $ make rudra-theano

To build Rudra with a mock learner (for testing purposes):

    $ make rudra-mock

The make variable `X10RTIMPL` chooses the implementation of 
[X10RT](http://x10-lang.org/documentation/x10rt.html). You can use whichever
versions of X10RT are supported on your platform e.g. sockets, pami, mpi.
(Note: mpi does not currently work on the IBM-internal dcc system.)

To build the default version of Rudra with X10RT for MPI, run:

    $ make rudra-theano X10RTIMPL=mpi

## Building Individual Components

To build librudra:

    $ cd cpp && make

To build the Theano learner:

    $ cd theano && make

Note:

1. `rudra.profile` sets the necessary environment variables needed for building and running Rudra. Amongst other things, it sets the `$RUDRA_HOME` environment variable. In some cases, you may need to modify `rudra.profile` to correctly point to your local Python installation. 

# Verifying the Build (Theano version)

For the Theano learner, because of the current limitation that only one NN model can be used per process, we have to use `-nwMode send_broadcast`, and also (for now) `-noTest` (testing is currently in-process and creates its own copy of the NN model). 

`send_broadcast` requires at least two processes (= X10 places) ... one serves as a parameter server. You set the number of places with an environment variable (`export X10_NPLACES=2`). 

Try running with mlp.py:

    $ export X10_NPLACES=2
    $ ./rudra -f examples/theano-mnist.cfg -nwMode send_broadcast -noTest -ll 0 -lr 0 -lt 0  -lu 0 

Log level 0 (TRACING) prints the maximum amount of information. If you don't want it, skip the -l* flags.

# Documentation

Dependencies: [Doxygen](http://www.stack.nl/~dimitri/doxygen/) v1.8.0 or higher

    $ cd doc && doxygen
