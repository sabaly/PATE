import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from data_loader import *
import numpy as np

def define_model(input_shape):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(16, input_shape=input_shape, activation="relu"),
        tf.keras.layers.Dense(32, input_shape=input_shape, activation="relu"),
        tf.keras.layers.Dense(64, input_shape=input_shape, activation="relu"),
        tf.keras.layers.Dense(128,input_shape=input_shape, activation="relu"),
        tf.keras.layers.Dense(64, input_shape=input_shape, activation="relu"),
        tf.keras.layers.Dense(32, input_shape=input_shape, activation="relu"),
        tf.keras.layers.Dense(16, input_shape=input_shape, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=[tf.keras.metrics.BinaryAccuracy(name="accuracy")])
    
    return model

def train_teacher(x_train, y_train, nb_epochs=20):
    model = st_model(x_train.shape[1:])

    model.fit(x_train, y_train, epochs = nb_epochs, versbose=False)

    return model

def train_teachers(subsets, nb_tchrs, nb_epochs=20):
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


def eval_teacher_model(model, x_test, y_test):
    print('Evaluation of a teacher model')
    eval1 = model.evaluate(x_test, y_test)
    print(f"**** Results \n\t-loss : {eval1[0]}\n\t-accuracy : {eval1[1]}")
    
