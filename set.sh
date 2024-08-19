#/bin/bash

for i in {1..10}
do
    mkdir "unit_$i"
    cd "unit_$i"
    for j in {1..2}
    do
        touch "session_$j.py"
    done
    cd ..
done
