# Rudra Distributed Learning Platform                         {#mainpage}

Rudra is a distributed framework for large-scale machine learning, which accepts training data and model configuration as inputs from the user and outputs the parameters of the trained model.

Detailed documentation on input formats, invocation, sample datasets and planned features can be found on the [project wiki](https://github.com/saraswat/rudra-dist/wiki).

# Installation

Dependencies:
1. g++ (4.4.7 or higher) / xlC
2. (optional - for Theano learner) [Theano](http://deeplearning.net/software/theano/)

To build librudra:

    $ cd cpp && make

To build Theano learner:

    $ cd theano && make

To build rudra-dist:

    $ cd x10 && make X10RTIMPL=sockets

Note:
1. `rudra.profile` sets the necessary environment variables needed for building and running Rudra. Amongst other things, it sets the `$RUDRA_HOME` environment variable. In some cases, you may need to modify `rudra.profile` to correctly point to your local Python installation. 

# Documentation

Dependencies: [Doxygen](http://www.stack.nl/~dimitri/doxygen/) v1.8.0 or higher

    $ cd doc && doxygen
