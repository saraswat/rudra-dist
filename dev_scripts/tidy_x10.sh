#/bin/bash
# this script tidies the performance log and can be reshaped by matlab to print out the pid/eid matrix
if [ $# -ne 3 ]; 
    then echo "illegal number of parameters "
    echo "tidy.sh log 15 11 --- e.g., log file name , 15 learners, and 11 events; output to log.analysis, the 12th event is actually eid=14 which is the applyUpdatemethod"
    exit
fi

fName=$1
pNum=$2  
eNum=$3
rm -fr $fName.analysis
## step 1 get the output file (from e1 to e 11)
for ((p=3;p<$((3+pNum));p++)) # 3 is the server number so learner rank starts from 3, for each learner
do
    for((e=1;e<=$eNum;e++)) # for each event
    do
	rm -fr ./_tmp.txt
	grep "P$p:E$e:" $1 | cut -d':' -f3 > ./_tmp.txt
	cat ./_tmp.txt | awk '{ sum+=$1} END {print sum}' >> $fName.analysis
    done
    rm -fr ./_tmp.txt
    grep "P$p:E14:" $1 | cut -d':' -f3 > ./_tmp.txt
    cat ./_tmp.txt | awk '{ sum+=$1} END {print sum}' >> $fName.analysis
done
rm -fr ./_tmp.txt

## step 2 get the output file (for e14)