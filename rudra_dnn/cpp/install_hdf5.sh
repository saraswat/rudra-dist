#!/bin/bash
# download and install HDF5
# creates a new directory $RUDRA_HOME/cpp/hdf5

DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
wget -N http://www.hdfgroup.org/ftp/HDF5/releases/hdf5-1.8.15/src/hdf5-1.8.15.tar.gz
tar -xvf hdf5-1.8.15.tar.gz
cp config.guess  ./hdf5-1.8.15/bin/
cd hdf5-1.8.15 hdf5

./configure --enable-cxx --prefix=$DIR/hdf5
make install

