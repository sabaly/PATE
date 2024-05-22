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

alpha = [100, 100]
update_alpha(alpha)
dataset = "acsemployment_bis"
nb_teachers = 10
st_train_times = 10
nb_fair_tchrs = 1 # wished

# prepare datasets !
subsets, student = get(dataset, nb_teachers, nb_fair_tchrs=nb_fair_tchrs)

# train teachers
teachers = parallel_training_teachers(subsets)
update_teachers(teachers)

accuracies, eod, spd, di = stats(nb_teachers, teachers, subsets)
set_metrics(eod)

nb_fair = [x < 0.1 for x in eod].count(True)

done = [0, 7]
if nb_fair  in done:
    print("OUPS ! >>> ", nb_fair, " <<<")
    exit(1)

if "_bis" in dataset:
    name = dataset + "_" + str(nb_fair) + "_fair_wv"+ ".png"
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

for cf in confs:
    # setting conf
    if cf != "All" :#and (nb_fair == 0 or nb_fair == nb_teachers):
        break
    print(f'Training  {cf} teachers')
    if cf == "All":
        aggregator = plurality
    elif cf == "Only fair":
        aggregator = only_fair
    else:
        if nb_fair == nb_teachers:
            continue
        aggregator = only_unfair
    y_train = np.asarray(aggregator(x_train))
    yhat_test = np.asarray(aggregator(x_test))
    y_axis = []
    print("Training students ... ", end="")
    for _ in range(st_train_times):
        st_model = train_student(x_train, y_train, verbose=False, nb_epochs=200)
        #eval_student_model(st_model, x_test, y_test, yhat_test, verbose=False)
        st_stats = fairness(st_model, x_test, yhat_test, s_test)
        y_axis.append(st_stats["EOD"])
    print("Done")
    ax2.plot(list(range(st_train_times)), y_axis, colors[color_index], label=cf)
    color_index = color_index + 1

aggregator = weighed_vote
y_train = np.asarray(aggregator(x_train))
yhat_test = np.asarray(aggregator(x_test))
y_axis = []
print("Training students ... ", end="")
for _ in range(st_train_times):
    st_model = train_student(x_train, y_train, verbose=False, nb_epochs=200)
    #y_pred = eval_student_model(st_model, x_test, y_test, yhat_test, verbose=False)
    st_stats = fairness(st_model, x_test, yhat_test, s_test)
    y_axis.append(st_stats["EOD"])
print("Done")
ax2.plot(list(range(st_train_times)), y_axis, colors[color_index], label="weighed vote", linestyle="dashed")

plt.title(f"PATE impacts on fairness")
plt.legend()
plt.savefig("../img/"+name)




