if [ "$#" -ne 3 ]
then
    echo "Wrong arguments. [Usage]: rudra_dcc_x10.sh config_file node_num job_id"
    exit -1
fi
: ${RUDRA_HOME:?"Need to set RUDRA_HOME"}

if [ ! -x $RUDRA_HOME/rudra ]
then
    echo "executable $RUDRA_HOME/rudra doesn't exist or it is not executable"
fi
conf_file=$1
node_num=$2
job_id=$3
if [ $node_num -gt 32 ] || [ $node_num -lt 1 ]
then 
    echo "User requested $node_num nodes: Rudra only supports 1-32 nodes."
    exit -1
fi

source $RUDRA_HOME/rudra.profile
export X10RT_MPI_THREAD_SERIALIZED=true
export OMP_NUM_THREADS=4
export X10_NTHREADS=2
rm -rf $RUDRA_HOME/LOG/$job_id # Rudra won't run if previous logs are in the way

jbsub -queue x86_excl -c ${node_num}xA.2 -out $job_id.out -err $job_id.err mpiwrap.sh $RUDRA_HOME/rudra -f ${conf_file} -j $job_id -ln 3 -ll 2 -lr 3 
