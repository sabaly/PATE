#!/bin/sh
nb_teachers="10 30 45"
for nb_tchrs in $nb_teachers
do
    echo ">>> Number of teachers : " $nb_teachers
    for nb_fair in $(seq 1 $(($nb_tchrs/2)))
    do
        python3 fairness_impact_eval.py $nb_tchrs $nb_fair
    done
done