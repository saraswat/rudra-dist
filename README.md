# Rudra Distributed Learning Platform                         {#mainpage}

Rudra is a distributed framework for large-scale machine learning, which accepts training data and model configuration as inputs from the user and outputs the parameters of the trained model.

Detailed documentation on input formats, invocation, sample datasets and planned features can be found on the [project wiki](https://github.com/saraswat/rudra-dist/wiki).

# Installation

Dependencies:
1. g++ (4.4.7 or higher) / xlC
2. (optional - for Theano learner) [Theano](http://deeplearning.net/software/theano/)

To build everything, just run:

    $ source rudra.profile
    $ make

To build librudra:

    $ cd cpp && make

To build Theano learner:

    $ cd theano && make

To build rudra-dist:

    $ cd x10 && make X10RTIMPL=sockets

Note:
1. `rudra.profile` sets the necessary environment variables needed for building and running Rudra. Amongst other things, it sets the `$RUDRA_HOME` environment variable. In some cases, you may need to modify `rudra.profile` to correctly point to your local Python installation. 

# Verifying that the basic Theano program runs

Try running with mlp.py:

    $ ./rudra -f examples/theano-mnist.cfg -ll 0 -lr 0 -lt 0  -lu 0 

Log level 0 (TRACING) prints the maximum amount of information. If you don't want it, skip the -l* flags.

# Documentation

Dependencies: [Doxygen](http://www.stack.nl/~dimitri/doxygen/) v1.8.0 or higher

    $ cd doc && doxygen
