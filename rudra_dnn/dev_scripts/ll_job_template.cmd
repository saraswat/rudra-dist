#@ output = ./std.out
#@ error = ./std.err
#@ node = 1
#@ tasks_per_node = 1
#@ job_type = parallel
#@ network.X10 = sn_single,not_shared,us
#@ checkpoint = no
#@ bulkxfer = yes
#@ collective_groups = 4
#@ class = day
#@ queue

export OMP_NUM_THREADS=4
export MP_SHARED_MEMORY=yes
#export JAVA_HOME=/opt/ibm/ibm-java-ppc64-71/jre
#export PATH=$PATH:/opt/ibmcmp/vacpp/12.1/bin/
export CHANGE_SMT=yes
export SMTMODE=4
poe ./Test -f config
