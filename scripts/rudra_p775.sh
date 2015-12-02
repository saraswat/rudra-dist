#!/bin/bash

if [ "$#" -ne 4 ]
then
    echo "Wrong arguments. [Usage]: rudra_p775.sh config_file node_num job_id sync_level"
    exit -1
fi

: ${RUDRA_HOME:?"Need to set RUDRA_HOME"}
source $RUDRA_HOME/rudra.profile
if [ ! -x $RUDRA_HOME/rudra ]
then
    echo "executable $RUDRA_HOME/rudra doesn't exist or it is not executable"
fi

: ${RUDRA_LEARNER_HOME:?"Need to set RUDRA_LEARNER_HOME"}

conf_file=$1
if [ ! -r "$conf_file" ];
then
    echo "$conf_file doesn't exist or it is not readable."
    exit -1
fi

node_num=$2
if [ $node_num -gt 32 ] || [ $node_num -lt 1 ]
then 
    echo "We only support to [1-32] nodes, you are asking for $node_num nodes."
    exit -1
fi

tasks_per_node=4

job_id=$3
nwMode=4 # by default apply
nwMode?=$4
tmpCmdFile=$job_id.cmd
rm -fr $tmpCmdFile

echo "#@ output = $job_id.stdout " >> $tmpCmdFile
echo "#@ error = $job_id.stderr " >> $tmpCmdFile
echo "#@ node = $node_num" >> $tmpCmdFile
echo "#@ tasks_per_node = $tasks_per_node" >> $tmpCmdFile
echo "#@ job_type = parallel" >> $tmpCmdFile
echo "#@ network.X10 = sn_single,not_shared,us" >> $tmpCmdFile
echo "#@ checkpoint = no" >> $tmpCmdFile
echo "#@ bulkxfer = yes" >> $tmpCmdFile
echo "#@ collective_groups = 4" >> $tmpCmdFile # disabled on May 8, 2015, as it seems p775 is not supporting it, at least temporarily.
echo "#@ class = day" >> $tmpCmdFile
echo "#@ queue" >> $tmpCmdFile
echo "source $RUDRA_HOME/rudra.profile" >> $tmpCmdFile
echo "source $RUDRA_LEARNER_HOME/rudra-learner.profile" >> $tmpCmdFile
echo "export OMP_NUM_THREADS=4" >> $tmpCmdFile
echo "export X10_NTHREADS=4" >> $tmpCmdFile
echo "export X10_NUM_IMMEDIATE_THREADS=1" >> $tmpCmdFile
echo "export MP_SHARED_MEMORY=yes" >> $tmpCmdFile
#echo "poe $RUDRA_HOME/rudra -f $conf_file -s adagrad -hard -ln 3 -ll 2 -lr 2 -j $job_id" >> $tmpCmdFile
echo "poe $RUDRA_HOME/rudra -f $conf_file -s adagrad -nwMode $nwMode -ln 3 -ll 2 -lr 2 -j $job_id" >> $tmpCmdFile
llsubmit $tmpCmdFile
