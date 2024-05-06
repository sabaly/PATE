import sys
from teachers import *
from student import *
from aggregator import *

import warnings

def fxn():
    warnings.warn("deprecated", DeprecationWarning)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fxn()


# define initial parameters  
# teachers dataset, nombers of teachers

if len(sys.argv) < 3:
    print("Usage : app.py <dataset_name> <number of teachers>")
    print("Default : dataset --> adult | number of teacher --> 250")
    datset = "adult"
    nb_teachers = 250
else:
    dataset = sys.argv[1]
    nb_teachers = int(sys.argv[2])

# train teachers
teachers, (x_test, y_test) = train_teachers(dataset, nb_teachers)
init_teachers(teachers)

# fake student train data
train_size = int(0.8*len(x_test))
x_train = x_test[:train_size]
x_test = x_test[train_size:]
true_y_test = y_test[train_size:]
# define aggregation methode
aggregator = agg_noisy_vote
while True:
    action = int(input("1. Update aggregator \t 2. Train student\n(0 exit)>>"))
    if action == 1:
        aggregator = update_aggregator()
    elif action==2:
        st_model = train_student(x_train, aggregator)
        eval_student_model(st_model, x_test, true_y_test, aggregator)
    else:
        exit(0)
