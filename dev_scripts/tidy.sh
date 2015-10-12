#/bin/bash
# this script tidies the performance log and can be reshaped by matlab to print out the pid/eid matrix
if [ $# -ne 3 ]; 
    then echo "illegal number of parameters "
    echo "tidy.sh log 15 11 --- log file name 15 learners and 11 events"
    exit
fi
fName=$1
pNum=$2  
eNum=$3
rm -fr $fName.analysis
rm -fr $fName.analysis.tmp
for ((p=3;p<$((3+pNum));p++)) # 3 is the server number so learner rank starts from 3
do
    for((e=1;e<=$eNum;e++))
    do
	rm -fr ./_tmp.txt
	grep "P$p:E$e:" $1 | cut -d':' -f3 > ./_tmp.txt
	cat ./_tmp.txt | awk '{ sum+=$1} END {print sum}' >> $fName.analysis.tmp
    done

done
sed 's/^$/0/' $fName.analysis.tmp > $fName.analysis # replace blank line by 0
rm -fr $fName.analysis.tmp
rm -fr ./_tmp.txt