if [ "$#" -ne 4 ]
then
    echo "Wrong arguments. [Usage]: rudra_dcc.sh config_file node_num job_id sync_level"
    exit -1
fi
: ${RUDRA_HOME:?"Need to set RUDRA_HOME"}

if [ ! -x $RUDRA_HOME/cpp/rudra ]
then
    echo "executable $RUDRA_HOME/cpp/rudra doesn't exist or it is not executable"
fi
conf_file=$1
node_num=$2
job_id=$3
sync_level=$4
sl=4 # by default smart
if [ $node_num -gt 32 ] || [ $node_num -lt 4 ]
then 
    echo "User requested $node_num nodes: Rudra only supports 4-32 nodes."
    exit -1
fi
if [ "$sync_level" == "hard" ] 
then
sl=0
elif [ "$sync_level" == "smart" ]
then 
sl=4
else
echo "$sync_level is not a legit sync level, legit sync level: hard|smart"
fi
source $RUDRA_HOME/rudra.profile
export OMP_NUM_THREADS=4
#jbsub  -c ${node_num}xA /opt/share/openmpi-1.8.4/bin/mpirun -n $node_num $RUDRA_HOME/cpp/rudra -f ${conf_file} -sl $sl -sc 3 -ll 1 -j $job_id

jbsub  -c ${node_num}x4 mpiwrap.sh $RUDRA_HOME/cpp/rudra -f ${conf_file} -sl $sl -sc 3 -ll 1 -j $job_id
