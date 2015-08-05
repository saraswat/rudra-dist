grep "Train" *.stdout  | tail -n 100 | cut -d':' -f3 | cut -d' ' -f1 | awk '{ sum += $1 } END { print sum }' 
