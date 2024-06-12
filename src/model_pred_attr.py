import sys, os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from student import *
from aggregator import *
import matplotlib.pyplot as plt
from analysis import *
import warnings
from teacher_ensemble import *

colors = plt.rcParams["axes.prop_cycle"].by_key()['color']
color_index = 0

def fxn():
    warnings.warn("deprecated", DeprecationWarning)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fxn()

nb_teachers = 20
nb_fair = 5


tchrs_ensemble = Ensemble(nb_teachers, nb_fair) 
# prepare datasets !

# teachers
update_teachers(tchrs_ensemble.tchrs)


# load student dataset
(x_train, x_test, y_train, y_test, s_train, s_test) = load_student_data("AK")

aggregator = plurality

labeled_y_train = aggregator(x_train, group=s_train)
labeled_t_test = aggregator(x_test, group=s_test)


