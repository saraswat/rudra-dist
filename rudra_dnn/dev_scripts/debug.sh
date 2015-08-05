start=1
end=150
learner=4
for ((c=$start; c<=$end; c++))
do
    for((i=1; i <=$learner; i++))
    do
	f1="E1_MB"$c"_U_LA_"$i".dump"
	f2="E1_MB"$c"_U_LA_"$i"_PS.dump"
	x= `diff $f1 $f2`
	if [ $? -ne 0 ]
	then
	    echo $f1 $f2 "diff"
	fi
	f1="E1_MB"$c"_B_LA_"$i".dump"
	f2="E1_MB"$c"_B_PS.dump"
	x= `diff $f1 $f2`
	if [ $? -ne 0 ]
	then
	    echo $f1 $f2 "diff"
	fi
     done
    #ls -l "E1_MB"$c"_U_LA_1.dump"
done