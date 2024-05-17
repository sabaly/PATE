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

alpha = [150, 50]
update_alpha(alpha)
dataset = "acsemployment_bis"
nb_teachers = 30
st_train_times = 50

if "_bis" in dataset:
    name = dataset + "_" + str(alpha[0]) + "_" + str(alpha[1]) + ".png"
else:
    name = dataset + ".png"

# prepare datasets !
subsets, student = get(dataset, nb_teachers)

# train teachers
teachers = train_teachers(subsets, nb_teachers)
update_teachers(teachers)

accuracies, eod, spd, di = stats(nb_teachers, teachers, subsets)
set_metrics(eod)


# load student dataset
(x_train, x_test, y_train, y_test, s_train, s_test) = load_student_data(student)


confs = ["All", "Only fair", "Only unfair"]

fig, (ax1, ax2)= plt.subplots(1, 2, sharey=True)
b_width = 0.3
x = range(len(accuracies))
# teachers hist 
ax1.bar(x, eod, width = b_width, color=[colors[3] for _ in eod],label="EOD")
cp_state = states.copy()
cp_state.pop(2)
ax1.set_xticks(x, ['' for _ in range(nb_teachers)])
ax1.set_yticks(np.arange(0, 1.1, step=0.1))
ax1.set_ylim([0,1.1])
ax1.set_xlabel("Teachers")
ax1.set_ylabel("Metrics")

for cf in confs:
    print(f'Training  {cf} teachers')
    # setting conf
    if cf == "All":
        aggregator = plurality
    elif cf == "Only fair":
        aggregator = only_fair
    else:
        aggregator = only_unfair
    
    y_axis = []
    for _ in range(st_train_times):
        st_model = train_student(x_train, aggregator)
        y_pred = eval_student_model(st_model, x_test, y_test, aggregator, verbose=False)
        st_stats = fairness(st_model, x_test, y_pred, s_test)
        y_axis.append(st_stats["EOD"])
    ax2.plot(list(range(st_train_times)), y_axis, colors[color_index], label=cf)
    color_index = color_index + 1

plt.title(f"PATE impacts on fairness")
plt.legend()
plt.savefig("../img/"+name)




