import sys
from teachers import *
from student import *
from aggregator import *
#import matplotlib as mpl
import matplotlib.pyplot as plt
from analysis import *
import warnings

colors = plt.rcParams["axes.prop_cycle"].by_key()['color']
color_index = 0

def fxn():
    warnings.warn("deprecated", DeprecationWarning)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fxn()


# define initial parameters  
# teachers dataset, nombers of teachers

if len(sys.argv) < 3:
    print("Usage : app.py <dataset_name> <number of teachers>")
    dataset = "adult"
    nb_teachers = 25
    print(f"Default : dataset --> {dataset} | number of teacher --> {nb_teachers}")
else:
    dataset = sys.argv[1]
    nb_teachers = int(sys.argv[2])

# prepare datasets !
subsets = get(dataset, nb_teachers)

# train teachers
teachers = train_teachers(subsets, nb_teachers)
init_teachers(teachers)

accuracies, eod, spd = stats(nb_teachers, teachers, subsets)

# statudent dataset
if dataset == "adult":
    # fake student train data
    x_test = None
    y_test = None
    for subset in subsets:
        if x_test is None:
            x_test = subset[1]
            y_test = subset[3]
        else:
            x_test = np.concatenate((x_test, subset[1]))
            y_test = np.concatenate((y_test, subset[3]))
    train_size = int(0.8*len(x_test))
    x_train = x_test[:train_size]
    x_test = x_test[train_size:]
    true_y_test = y_test[train_size:]
else:
    # load student dataset
    pass
# define aggregation methode
aggregator = agg_noisy_vote
while True:
    action = int(input("1. Update aggregator \t 2. Train student \t 3. Stats\n(0 exit)>>"))
    if action == 1:
        aggregator = update_aggregator()
    elif action==2:
        st_model = train_student(x_train, aggregator)
        eval_student_model(st_model, x_test, true_y_test, aggregator)
    elif action==3:
        fig, tchr_ax = plt.subplots()
        b_width = 0.3
        x1 = range(len(accuracies))
        x2 = [x + b_width for x in x1]
        x3 = [x + b_width for x in x2]
        #print(accuracies, "\n", eod, "\n", spd)
        tchr_ax.bar(x1, accuracies, width = b_width, color=colors[0], label="accuracy")
        tchr_ax.bar(x2, eod, width = b_width, color=[colors[1] for _ in eod],label="EOD")
        tchr_ax.bar(x3, spd, width = b_width, color=[colors[2] for _ in spd], label="SPD")
        tchr_ax.set_xticks([x + b_width/4 for x in x2], [t+1 for t in range(nb_teachers)])
        tchr_ax.set_yticks(np.arange(0, 1.1, step=0.1))
        tchr_ax.set_ylim([0,1.1])
        tchr_ax.set_xlabel("Teachers")
        tchr_ax.set_ylabel("Metrics")
        plt.legend()
        plt.show()
    else:
        exit(0)