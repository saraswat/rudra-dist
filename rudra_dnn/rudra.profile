DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
export RUDRA_HOME=$DIR

# CMU ParameterServer
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$RUDRA_HOME/parameter_server/third_party/lib

# HDF5 lib
export HDF5_INCLUDE_PATH=$RUDRA_HOME/cpp/hdf5/include
export HDF5_LIB_PATH=$RUDRA_HOME/cpp/hdf5/lib
export LD_LIBRARY_PATH=$HDF5_LIB_PATH:$LD_LIBRARY_PATH

# OpenBLAS
if [ x$OPENBLAS_LIB_PATH == "x"  ] ## set OPENBLAS_LIB_PATH only when it is not set
then
	export OPENBLAS_INCLUDE_PATH=~/opt/OpenBLAS/include
	export OPENBLAS_LIB_PATH=~/opt/OpenBLAS/lib
fi
export LD_LIBRARY_PATH=$OPENBLAS_LIB_PATH:$LD_LIBRARY_PATH

#ATLAS:
export ATLAS_LIB_PATH=/usr/lib64/atlas
export LD_LIBRARY_PATH=$ATLAS_LIB_PATH:$LD_LIBRARY_PATH
# profiler
# with xlC on BigData
export LD_LIBRARY_PATH=/gpfs/my_gpfs/home/jjmiltho/gperftools/lib:$LD_LIBRARY_PATH

export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

# googletest - run cpp/install_gtest.sh to install
export GTEST_DIR=$RUDRA_HOME/cpp/gtest-1.7.0

#Add mpicxx to path (needed for deep computing cluster)
export MPICXX_PATH=/opt/share/openmpi-1.8.4/bin
export PATH=$MPICXX_PATH:$PATH

# If Cuda is avaible on this machine then you need to set CUDA_PATH to the desired Cuda version.
# If CUDA_PATH is set and it looks correct then the variable RUDRA_CUDA is set here
# and used by Makefile as indication that should compile Cuda programs
if [ -z $CUDA_PATH ] || [ ! -x $CUDA_PATH/bin/nvcc ]
then
        # Cuda not wanted or nvcc not available
        unset RUDRA_CUDA
else
        # Looks OK. Set the variables
        export RUDRA_CUDA=yes
        export PATH=$PATH:$CUDA_PATH/bin
        export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDA_PATH/lib64
fi        
                
        

