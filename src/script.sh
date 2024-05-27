#!/bin/sh
nb_teachers="30"
for nb_tchrs in $nb_teachers
do
    echo ">>> Number of teachers : " $nb_tchrs
    for nb_fair in $(seq 20 $nb_tchrs) #$((4 * $nb_tchrs / 5))
    do
        python3 fairness_impact_eval.py $nb_tchrs $nb_fair
    done
done