if [ "$#" -ne 3 ]
then
    echo "Wrong arguments. [Usage]: rudra_dcc_x10.sh config_file node_num job_id"
    exit -1
fi
: ${RUDRA_HOME:?"Need to set RUDRA_HOME"}

if [ ! -x $RUDRA_HOME/x10/rudra ]
then
    echo "executable $RUDRA_HOME/x10/rudra doesn't exist or it is not executable"
fi
conf_file=$1
node_num=$2
job_id=$3
if [ $node_num -gt 32 ] || [ $node_num -lt 2 ]
then 
    echo "User requested $node_num nodes: Rudra only supports 2-32 nodes."
    exit -1
fi

source $RUDRA_HOME/rudra.profile
export OMP_NUM_THREADS=4
export X10_NTHREADS=2
export X10_NUM_IMMEDIATE_THREADS=1
rm -rf $RUDRA_HOME/LOG/$job_id # Rudra won't run if previous logs are in the way

jbsub  -c ${node_num}x16.2 -out $job_id.out -err $job_id.err mpiwrap.sh $RUDRA_HOME/x10/rudra -f ${conf_file} -j $job_id -ln 3 -ll 2 -lr 2
