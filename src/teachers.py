import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from data_loader import *
import numpy as np
import multiprocessing as mp
from multiprocessing import Pool

def define_model(input_shape, index=0):
    tf.keras.utils.set_random_seed(index)
    model = tf.keras.models.Sequential([
        tf.keras.Input(input_shape),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=[tf.keras.metrics.BinaryAccuracy(name="accuracy"), tf.keras.metrics.Recall(name="recall")])
    
    return model

def train_teachers(subsets, nb_tchrs, nb_epochs=60):
    fi = []

    i = 0
    for subset in subsets:
        x_train, _, y_train, _, _, _ = subset
        x_train, y_train = np.array(x_train), np.array(y_train)

        model = define_model((x_train.shape[-1],))

        model.fit(x_train, y_train, epochs=nb_epochs, verbose=False)
        fi.append(model)
        pg = int(i/nb_tchrs * 100)
        i+=1
        print(f"\rTraining Teachers: {pg}%", end="")
    print(f"\rTraining Teachers: 100%")

    return fi

def wrapper(args):
    return train_teacher(*args)

def train_teacher(subset, index, nb_epochs=60):
    x_train, _, y_train, _, _, _ = subset
    x_train, y_train = np.array(x_train), np.array(y_train)

    model = define_model((x_train.shape[-1],), index=index)
    model.fit(x_train, y_train, epochs=nb_epochs, verbose=False)

    return model
    
def parallel_training_teachers(subsets):
    print("Training teachers...", end="")
    with Pool(mp.cpu_count()) as p:
        fi = p.map(wrapper, [(subsets[i], i) for i in range(len(subsets))])
    print("Done")
    return fi

def eval_teacher_model(model, x_test, y_test):
    print('Evaluation of a teacher model')
    eval1 = model.evaluate(x_test, y_test)
    print(f"**** Results \n\t-loss : {eval1[0]}\n\t-accuracy : {eval1[1]}")
    

def aggregate_dataset(id, features, labels, group):
    p_grp_tp = features[(group == 1) & (labels == 1)]
    up_grp_tp = features[(group == 2) & (labels == 1)]
    S = pd.concat([p_grp_tp, up_grp_tp])
    TCHRID = [id]*S.shape[0]
    S = S.assign(TCHRID=TCHRID)
    return S

def global_fair_component(id, yhat, S, y_test, s_test):
    sum_ni = S.shape[0]
    n_id = S[S["TCHRID"]==id].shape[0]
    a = np.mean(yhat[(y_test == 1) & (s_test == 1) & (S["TCHRID"] == id)])
    b = np.mean(S[(s_test == 1) & (y_test ==1)][S["TCHRID"] == id])
    c = np.mean(S[(s_test == 1) & (y_test == 1)])
    a = 0 if a is None else a
    b = 0 if b is None else b
    c = 0 if c is None else c
    p_grp = a*b/c
    a = np.mean(yhat[(y_test == 1) & (s_test == 2) & (S["TCHRID"] == id)])
    b = np.mean(S[(s_test == 2) & (y_test ==1)][S["TCHRID"] == id])
    c = np.mean(S[(s_test == 2) & (y_test == 1)])
    a = 0 if a is None else a
    b = 0 if b is None else b
    c = 0 if c is None else c
    up_grp = a*b/c
    m = up_grp - p_grp

    return m



