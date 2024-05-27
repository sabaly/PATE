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

if len(sys.argv) < 3:
    print("Usage : fairness_impact_eval.py <nb_teachers> <nb_fair_wished>")
    exit(1)

dataset = "acsemployment_bis"
nb_teachers = int(sys.argv[1])
nb_fair_tchrs = int(sys.argv[2]) # wished

# prepare datasets !
subsets, student = get(dataset, nb_teachers, nb_fair_tchrs=nb_fair_tchrs)

# train teachers
teachers = parallel_training_teachers(subsets)
update_teachers(teachers)

accuracies, eod, spd, di = stats(nb_teachers, teachers, subsets)
set_metrics(eod)

nb_fair = [x < 0.1 for x in eod].count(True)

""" if abs(nb_fair_tchrs - nb_fair) > 2:
    exit(1) """
if "_bis" in dataset:
    name = dataset + "_" + str(nb_fair) + "_fair"+ ".png"
else:
    name = dataset + ".png"
    nb_fair = 0

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

#eods = []
for cf in confs:
    # setting conf
    if cf != "All" and (nb_fair == 0 or nb_fair == nb_teachers):
        break
    print(f'Training  {cf} teachers')
    if cf == "All":
        aggregator = plurality
        x = 1
    elif cf == "Only fair":
        aggregator = only_fair
        x =1 + b_width
    else:
        if nb_fair == nb_teachers:
            continue
        aggregator = only_unfair
        x =1 + 2*b_width
    y_train = np.asarray(aggregator(x_train))
    yhat_test = np.asarray(aggregator(x_test))
    st_model = train_student(x_train, y_train, verbose=False)
    ev1, ev2 = eval_student_model(st_model, x_test, y_test, yhat_test)
    st_stats = fairness(st_model, x_test, yhat_test, s_test)
    #ax2.plot([0, 1], [ev2[1]])
    ax2.bar([x], st_stats["EOD"], width = b_width, color=colors[color_index],label=f"{cf} [acc - {int(ev1[1]*100), int(ev2[1]*100)}]")
    color_index += 1

""" 
aggregator = weighed_vote
y_train = np.asarray(aggregator(x_train))
yhat_test = np.asarray(aggregator(x_test))

st_model = train_student(x_train, y_train, verbose=False)
st_stats = fairness(st_model, x_test, yhat_test, s_test)

ax2.bar([1 + 3*b_width], st_stats["EOD"], width = b_width, color=colors[3], label="weighed vote")
"""
#ax2.set_xticks([1 + b_width], ['']) 

plt.title(f"PATE impacts on fairness")
plt.legend()
path = "../img/archive_" + str(nb_teachers) + "/"
plt.savefig(path+name)


