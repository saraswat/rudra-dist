DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
export RUDRA_HOME=$DIR
export RUDRA_LIB=$RUDRA_HOME/lib
export RUDRA_INCLUDE=$RUDRA_HOME/include

export LD_LIBRARY_PATH=$RUDRA_LIB:$LD_LIBRARY_PATH
export PYTHONPATH=$PYTHONPATH:$RUDRA_HOME/theano

# Python, numpy and Theano
export LIBRARY_PATH=$LIBRARY_PATH:/opt/share/Python-2.7.9/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/local/cuda/lib:/opt/share/Python-2.7.9/lib/
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/share/cuda-7.0/lib64:/opt/share/cuda-7.0/lib:/opt/share/Python-2.7.9/lib/
export PATH=/opt/share/Python-2.7.9/bin/:$PATH

# googletest - run cpp/install_gtest.sh to install
export GTEST_DIR=$RUDRA_HOME/cpp/gtest-1.7.0

