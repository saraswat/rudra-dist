#/bin/bash
# this scripts takes a lit of Rudra OUT file and plot the accuracy vs time
if [ $# -lt 1 ];
    then echo "[Usage] acc_time_plot.sh phase1.OUT phase2.OUT phase3.OUT ..."
    exit
fi
results=()
for arg in "$@"
do
    result=$arg.accdat
    results+=($result)
    rm -fr $result
    grep "Epoch" -A1000 "$arg" | sed 's/#//g' | tail -n +2 > "$result"
done
rm -fr ./tmp.accdat
python $RUDRA_HOME/dev_scripts/prep_acc_time.py ${results[@]} > ./tmp.accdat
gnuplot -e "accfile='./tmp.accdat'" $RUDRA_HOME/dev_scripts/acc_time.gnuplot
