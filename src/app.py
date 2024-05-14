import sys
from teachers import *
from student import *
from aggregator import *
from data_loader import load_student_data
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
list_dataset = {"adult", "acsemployment"}
if len(sys.argv) < 3:
    print("Usage : app.py <dataset_name> <number of teachers>")
    dataset = "adult"
    nb_teachers = 25
    print(f"Default : dataset --> {dataset} | number of teacher --> {nb_teachers}")
elif sys.argv[1] in list_dataset:
    dataset = sys.argv[1]
    nb_teachers = int(sys.argv[2])
else:
    print(f"{sys.argv[1]} is not available !")
    exit(1)

# prepare datasets !
subsets, student = get(dataset, nb_teachers)

# train teachers
teachers = train_teachers(subsets, nb_teachers)
init_teachers(teachers)

accuracies, eod, spd, di = stats(nb_teachers, teachers, subsets)
set_metrics(eod)

# statudent dataset
if student == None:
    # fake student train data
    x_test = None
    y_test = None
    s_test = None
    for subset in subsets:
        if x_test is None:
            x_test = subset[1]
            y_test = subset[3]
            s_test = subset[5]
        else:
            x_test = np.concatenate((x_test, subset[1]))
            y_test = np.concatenate((y_test, subset[3]))
            s_test = np.concatenate((s_test, subset[5]))
    train_size = int(0.8*len(x_test))
    x_train = x_test[:train_size]
    x_test = x_test[train_size:]
    y_test = y_test[train_size:]
    s_test = s_test[train_size:]
else:
    # load student dataset
    (x_train, x_test, y_train, y_test, s_train, s_test) = load_student_data(student)
# default aggregation methode
aggregator = plurality
student_trained = False
while True:
    action = int(input("1. Update aggregator \t 2. Train student \t3. Set number fair Teachers\n 4. Stats\n(0 exit)>> "))
    if action == 1:
        aggregator = update_aggregator(aggregator)
    elif action==2:
        st_model = train_student(x_train, aggregator)
        y_pred = eval_student_model(st_model, x_test, y_test, aggregator)
        st_stats = fairness(st_model, x_test, y_pred, s_test)
        student_trained = True
    elif action==3:
        nb_fair_tchrs = int(input("Enter number of fair teachers >>> "))
        subsets, student = get(dataset, nb_teachers=nb_fair_tchrs, nb_fair_tchrs=nb_fair_tchrs)
        up_tchrs = train_teachers(subsets, nb_fair_tchrs)
        for i in range(nb_fair_tchrs):
            teachers[i] = up_tchrs[i]
        init_teachers(teachers)
    elif action==4:
        fig, (tchr_ax, st_ax)= plt.subplots(2, 1, sharey=True)
        b_width = 0.3
        x1 = range(len(accuracies))
        x2 = [x + b_width for x in x1]
        x3 = [x + b_width for x in x2]
        #x4 = [x + b_width for x in x3]
        # teachers hist 
        tchr_ax.bar(x1, accuracies, width = b_width, color=colors[0], label="accuracy")
        tchr_ax.bar(x2, eod, width = b_width, color=[colors[1] for _ in eod],label="EOD")
        tchr_ax.bar(x3, spd, width = b_width, color=[colors[2] for _ in spd], label="SPD")
        #tchr_ax.bar(x4, di, width = b_width, color=[colors[3] for _ in spd], label="DI")
        tchr_ax.set_xticks([x + b_width/4 for x in x2], [t+1 for t in range(nb_teachers)])
        tchr_ax.set_yticks(np.arange(0, 1.1, step=0.1))
        tchr_ax.set_ylim([0,1.1])
        tchr_ax.set_xlabel("Teachers")
        tchr_ax.set_ylabel("Metrics")

        # student hist
        if student_trained:
            st_ax.bar([1], [st_stats["ACC"]], width=b_width, color=colors[0], label="accuracy")
            st_ax.bar([1+b_width], [st_stats["EOD"]], width=b_width, color=colors[1], label="EOD")
            st_ax.bar([1+2*b_width], [st_stats["SPD"]], width=b_width, color=colors[2], label="SPD")
            st_ax.bar([1 + 3*b_width], [st_stats["DI"]], width = b_width, color=colors[3], label="DI")
            st_ax.set_xticks([1], ["student"])
            st_ax.set_yticks(np.arange(0, 1.1, step=0.1))
            st_ax.set_ylim([0,1.1])
            st_ax.set_xlabel(f"Student : {student}")
            st_ax.set_ylabel("Metrics")
        plt.legend()
        plt.show()
    else:
        exit(0)