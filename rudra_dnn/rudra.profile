DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
export RUDRA_HOME=$DIR

export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

# googletest - run cpp/install_gtest.sh to install
export GTEST_DIR=$RUDRA_HOME/cpp/gtest-1.7.0
        

